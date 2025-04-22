import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from datetime import datetime
from util import get_logger
from gru import BiGRUModel
from npy_dataloader import load_data


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    train_total = 0
    all_labels = []
    all_predicted = []

    progress_bar = tqdm(data_loader, desc='Training', leave=True)
    for len_data, time_data, batch_labels in progress_bar:
        len_data, time_data, batch_labels = len_data.long().to(device), time_data.float().to(device), batch_labels.long().to(device)
        outputs = model(len_data, time_data)

        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len_data.size(0)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(batch_labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())
        train_total += batch_labels.size(0)

    train_loss /= train_total

    accuracy = accuracy_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted, labels=[0, 1], pos_label=1)
    recall = recall_score(all_labels, all_predicted, labels=[0, 1], pos_label=1)
    f1 = f1_score(all_labels, all_predicted, labels=[0, 1], pos_label=1)

    return train_loss, accuracy, precision, recall, f1


def evaluate(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating', leave=True)
        for len_data, time_data, batch_labels in progress_bar:
            len_data, time_data, batch_labels = len_data.long().to(device), time_data.float().to(device), batch_labels.long().to(device)
            outputs = model(len_data, time_data)
            loss = criterion(outputs, batch_labels)

            test_loss += loss.item() * len_data.size(0)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(batch_labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            test_total += batch_labels.size(0)

    test_loss /= test_total

    accuracy = accuracy_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted, labels=[0, 1], pos_label=1)
    recall = recall_score(all_labels, all_predicted, labels=[0, 1], pos_label=1)
    f1 = f1_score(all_labels, all_predicted, labels=[0, 1], pos_label=1)

    return test_loss, accuracy, precision, recall, f1


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, logger):
    for epoch in range(num_epochs):
        start_time = time.time()

        # Adjust learning rate (optional)
        lr = optimizer.param_groups[0]['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)

        elapsed_time = time.time() - start_time

        logger.info(f'Epoch {epoch + 1}/{num_epochs} - Time: {elapsed_time:.2f}s - '
              f'Train Loss: {train_loss:.5f} - Train Accuracy: {train_accuracy:.5f} - '
              f'Train Precision: {train_precision:.5f} - Train Recall: {train_recall:.5f} - '
              f'Train F1: {train_f1:.5f} - Test Loss: {test_loss:.5f} - '
              f'Test Accuracy: {test_accuracy:.5f} - Test Precision: {test_precision:.5f} - '
              f'Test Recall: {test_recall:.5f} - Test F1: {test_f1:.5f}')


def run_training_experiment(KK1, config, logger, device):
    train_loader, test_loader = load_data(sample_frequency=config['sample_frequency'],
                                            batch_size=config['batch_size'], k=config['pkt_num'])

    model = BiGRUModel(input_size=config['input_size'],
                       hidden_size=config['hidden_size'],
                       num_layers=config['num_layers'],
                       te_shape=(config['batch_size'], config['pkt_num'], config['sample_frequency']),
                       droprate=config['droprate']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Run training and evaluation
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion,
                       num_epochs=config['num_epochs'], device=device, logger=logger)


def run(KK1):
    config = {
        'sample_frequency': 50,
        'input_size': 96,
        'pkt_num': 60,
        'hidden_size': 160,
        'num_layers': 4,
        'droprate': 0.5,
        'learning_rate': 0.0005,
        'batch_size': 160,
        'num_epochs': 70,
    }

    formatted_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), 'log', f'PackL_{KK1}_{formatted_start_time}')
    os.makedirs(log_path, exist_ok=True)
    logfile = os.path.join(log_path, 'result.csv')

    logger = get_logger(log_path, logfile)
    logger.info(config)

    run_training_experiment(KK1, config, logger, device)


if __name__ == '__main__':
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for kkk in range(0, 5):
        run(kkk)
