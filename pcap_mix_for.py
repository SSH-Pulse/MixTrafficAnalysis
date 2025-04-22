from scapy.all import rdpcap, wrpcap
import os
import numpy as np


def is_tcp_packet(pkt):
    """
    Checks if the packet is an IP packet with a TCP layer.
    """
    try:
        pkt['IP']
    except KeyError:
        return False  # Drop the packet if it's not an IP packet
    return "TCP" in pkt


def twopcapmix(down_pcap, up_pcap):
    """
    Merges two pcap files by packet time order.
    """
    i, j = 0, 0
    res_mix = []
    while i < len(down_pcap) or j < len(up_pcap):
        if i >= len(down_pcap):  # If all packets from down_pcap have been processed
            res_mix.append(up_pcap[j])
            j += 1
        elif j >= len(up_pcap):  # If all packets from up_pcap have been processed
            res_mix.append(down_pcap[i])
            i += 1
        else:
            p1, p2 = down_pcap[i], up_pcap[j]
            if p1.time <= p2.time:
                res_mix.append(p1)
                i += 1
            else:
                res_mix.append(p2)
                j += 1
    return res_mix


def twopcapmix_ssh(down_pcap, up_pcap):
    """
    Merges two pcap files with special handling for SSH packets.
    """
    i, j = 0, 0
    res_mix, i_record = [], []
    while i < len(down_pcap) or j < len(up_pcap):
        if i >= len(down_pcap):  # If all packets from down_pcap have been processed
            res_mix.append(up_pcap[j])
            i_record.append(0)  # Indicate that SSH packet came from up_pcap
            j += 1
        elif j >= len(up_pcap):  # If all packets from up_pcap have been processed
            res_mix.append(down_pcap[i])
            i_record.append(1)  # Indicate that SSH packet came from down_pcap
            i += 1
        else:
            p1, p2 = down_pcap[i], up_pcap[j]
            if p1.time <= p2.time:
                res_mix.append(p1)
                i_record.append(1)
                i += 1
            else:
                res_mix.append(p2)
                i_record.append(0)
                j += 1
    return res_mix, i_record


def extract_source_port(packet):
    """
    Extracts the source port from a TCP packet.
    """
    return packet['TCP'].sport if 'TCP' in packet else None


def extract_features(pcap):
    """
    Extracts features such as packet length and time delta from a pcap file.
    """
    features = {"packet_length": [], "arrive_time_delta": []}
    prev_time = None
    for pkt in pcap:
        if prev_time is None:
            features["arrive_time_delta"].append(0.0)
        else:
            features["arrive_time_delta"].append(float(pkt.time - prev_time))
        prev_time = pkt.time

        length = len(pkt)
        sport_packet = extract_source_port(pkt)
        if sport_packet is not None:
            if sport_packet <= 2225:
                features["packet_length"].append(length)
            else:
                features["packet_length"].append(-length)
        else:
            print("Invalid packet with no source port:", pkt)

    return features["packet_length"], features["arrive_time_delta"]


def merge_pcap_files(folder1_path, folder2_path, save_path):
    """
    Merges pcap files from two directories and saves the results in the specified directory.
    """
    pcap_list1 = ['ssh.pcap', '7.pcap', '6.pcap', '5.pcap', '4.pcap', '3.pcap', '2.pcap', '1.pcap']
    pcap_list2 = ['1.pcap', '2.pcap', '3.pcap', '4.pcap', '5.pcap', '6.pcap', '7.pcap', 'ssh.pcap']
    os.makedirs(save_path, exist_ok=True)

    for pcap_1 in pcap_list1:
        for pcap_2 in pcap_list2:
            if pcap_1 == pcap_2:  # Skip matching pcap files
                continue

            # Skip processing if the filenames match
            pcap1_name = pcap_1.split(".")[0]
            pcap2_name = pcap_2.split(".")[0]
            file_name = pcap1_name + pcap2_name

            print(f"{file_name} start mix!")

            pcap1_path = os.path.join(folder1_path, pcap_1)
            pcap2_path = os.path.join(folder2_path, pcap_2)
            packets1, packets2 = rdpcap(pcap1_path), rdpcap(pcap2_path)

            if pcap_1 == 'ssh.pcap':
                res_mix, i_record = twopcapmix_ssh(packets1, packets2)
                mix_rate_path = os.path.join(save_path, f'{file_name}_rate.npy')
                np.save(mix_rate_path, np.array(i_record))
            else:
                res_mix = twopcapmix(packets1, packets2)

            mix_packets_path = os.path.join(save_path, f'{file_name}.pcap')
            mix_length_path = os.path.join(save_path, f'{file_name}_length.npy')
            mix_time_path = os.path.join(save_path, f'{file_name}_time.npy')

            fea_length, fea_time = extract_features(res_mix)
            np.save(mix_length_path, np.array(fea_length))
            np.save(mix_time_path, np.array(fea_time))

            wrpcap(mix_packets_path, res_mix)

            print(f"{file_name} feature saving completed!")
            break  # Only process the first pair for now


# Example usage:
# folder_path1 = "/path/to/pcap1"
# folder_path2 = "/path/to/pcap2"
# output_file = "/path/to/output.pcap"
# merge_pcap_files(folder_path1, folder_path2, output_file)
