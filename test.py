from scapy.all import rdpcap
packets = rdpcap("raw\Friday-WorkingHours.pcap")
print(packets[0].time)  # unix timestamp