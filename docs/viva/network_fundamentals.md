# 🛜 Topic 2: Network Fundamentals & Column Reference
### Ultimate In-Depth Viva Study Guide: Network Anomaly Detection (CICIDS2017)

This document provides a comprehensive analysis of **Networking Fundamentals** and serves as a detailed reference for **every single column** in the CICIDS2017 dataset, mapping their technical meanings, network roles, and direct relevance to cyberattacks.

---

## 1. The 7 Layers of the OSI Model: Theory & Project Context

To impress your examiner, you must explain the **Open Systems Interconnection (OSI) Model** first as a general standard, and then demonstrate exactly how it maps to the **attacks, features, and algorithms** in this specific project.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                          THE 7-LAYER OSI MODEL                           │
├─────────────────┬─────────────────────────┬──────────────────────────────┤
│ Layer           │ General Concept         │ Project / NIDS Context       │
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ 7. Application  │ End-user applications   │ HTTP, SSH, FTP, Web Attacks, │
│                 │ (HTTP, DNS, SSH, FTP)   │ Brute Force logins, Botnets  │
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ 6. Presentation │ Formatting, encryption  │ SSL/TLS encryption. Bypassed │
│                 │ (SSL/TLS, ASCII, compression)│ via L3/L4 statistical checks│
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ 5. Session      │ Establish/manage sockets│ Sockets. Exploited by        │
│                 │ (APIs, connection sessions)│ Slowloris holding open connections│
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ 4. Transport    │ End-to-end ports & state│ Ports, TCP/UDP handshakes,   │
│                 │ (TCP, UDP, ports)       │ PortScans, SYN/RST flag features│
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ 3. Network      │ Host-to-host routing    │ IP addresses, packet counts, │
│                 │ (IP packets, routers)   │ DoS IP spoofing, ICMP floods │
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ 2. Data Link    │ Node-to-node frames     │ Ethernet frames, MAC         │
│                 │ (MAC addresses, switches)│ addresses (bypassed for ML)  │
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ 1. Physical     │ Raw bit transmission    │ Network interface card (NIC),│
│                 │ (Cables, fiber, SPAN port)│ raw PCAP tap via SPAN/mirror │
└─────────────────┴─────────────────────────┴──────────────────────────────┘
```

---

### Layer 1: The Physical Layer
* **General Theory**: Transmits raw, unstructured bitstreams over physical transmission media (ethernet cables, fiber optic lines, electromagnetic radio waves, and interface pins). It deals with hardware voltage levels, bit timings, and physical card connectors.
* **Project Context**: 
  * **The Tap**: In a live deployment of your system, a **Network Tap** or a **SPAN (Switch Port Analyzer) Port mirroring** configuration copies electromagnetic signals at the physical cable level and forwards them to your monitoring node.
  * **Hardware**: This is represented by the Physical Network Interface Card (NIC) collecting electrical impulses before they are grouped into software packets.

---

### Layer 2: The Data Link Layer
* **General Theory**: Groups L1 bitstreams into structured data chunks called **Frames**. It manages node-to-node (hop-to-hop) communication, utilizes physical **MAC (Media Access Control) Addresses**, and performs basic error detection and flow pacing.
* **Project Context**: 
  * **Frames**: Standard enterprise traffic runs over the **Ethernet Protocol** at Layer 2.
  * **ML Decision**: Why are MAC addresses ignored by your ML model? MAC addresses change at every router hop. If your machine learning model learned specific MAC address signatures, it would completely fail to generalize outside the local network. Your flow preprocessor discards L2 information to ensure generalizability.

---

### Layer 3: The Network Layer
* **General Theory**: Directs logical routing and host-to-host path determination across multiple independent networks. It packages frames into **Packets** and relies on **IP (Internet Protocol) Addresses** (IPv4/IPv6).
* **Project Context**:
  * **IP Spoofing**: In **DoS/DDoS floods (such as GoldenEye/Hulk)**, attackers spoof thousands of source IP addresses at Layer 3 to prevent the victim firewall from blocking a single malicious source.
  * **Volume Features**: Your model evaluates L3 volumes. Features tracking total packets, total byte counts, and subflow distributions are calculated based on L3 IP headers.
  * **ICMP Attacks**: Ping sweeps and ICMP flooding operate strictly at Layer 3.

---

### Layer 4: The Transport Layer
* **General Theory**: Manages end-to-end reliability, session multiplexing, error-checking, and flow control. It packages packets into **TCP Segments** or **UDP Datagrams**. It introduces **Ports** to identify which software application is communicating.
* **Project Context**:
  * **TCP Handshake**: TCP uses a three-way handshake (SYN $\to$ SYN-ACK $\to$ ACK) to establish connections. **SYN Floods (DDoS)** target this handshake, sending millions of SYN packets and leaving connections half-open to crash the server.
  * **Reconnaissance**: **PortScans** target L4 by systematically sending probes to ports ($1$ to $65535$) to spot listening services.
  * **L4 Features**: The core features in your dataset—like `SYN Flag Count`, `RST Flag Count`, `Init_Win_bytes_forward`, and `Destination Port`—operate strictly at Layer 4.

---

### Layer 5: The Session Layer
* **General Theory**: Manages connection establishment, maintenance, and teardown cycles (sessions) between local and remote applications. It handles socket synchronization and data checkpoints (e.g. NetBIOS, socket APIs).
* **Project Context**:
  * **Slowloris DoS**: This attack targets Layer 5 sessions. The attacker opens multiple connections to a web server and sends partial HTTP headers extremely slowly. The web server holds these L5 sockets open indefinitely, waiting for completion. This eventually exhausts the server's session pool, crashing it.
  * **Timing Features**: These session holds are flagged by your model through statistical features like `Flow Duration`, `Active Max/Min`, and `Idle Max/Min`.

---

### Layer 6: The Presentation Layer
* **General Theory**: Acts as the translator for data. It handles data formatting, compression, character conversion (e.g., ASCII/EBCDIC), and critical **encryption/decryption** standards (SSL/TLS).
* **Project Context**:
  * **SSL/TLS Encryption**: Secure applications use SSL/TLS at Layer 6 to encrypt payloads (HTTPS on port 443).
  * **The ML Advantage**: Deep Packet Inspection (DPI) firewalls look inside application payloads to spot malicious code, but they completely fail if the traffic is encrypted at Layer 6. **Your flow-based ML model bypasses this limitation** by ignoring payloads and looking strictly at L3/L4 statistical timing shapes, flagging encrypted Botnets or DDoS traffic without needing to decrypt L6.

---

### Layer 7: The Application Layer
* **General Theory**: The user-facing software layer where applications interact with network services. It includes high-level protocols like **HTTP/HTTPS (Web)**, **SSH (Secure Shell)**, **FTP (File Transfer)**, and **DNS (Domain Name System)**.
* **Project Context**:
  * **Web Attacks**: SQL Injection (SQLi) and Cross-Site Scripting (XSS) directly target Layer 7 web servers (Apache, Nginx running HTTP/HTTPS).
  * **Brute Force**: Targets L7 authentication mechanisms on FTP (Port 21) or SSH (Port 22).
  * **Botnets**: Command & Control (C2) communication operates over L7 protocols (like HTTP queries or DNS lookups) to blend in.
  * **NIDS Identification**: While these attacks occur at Layer 7, your model detects their L7 footprint through L3/L4 timing patterns, packet length variances, and flag states (e.g. Brute Force spiking RST flags, SQLi skewing average packet sizes).

---

## 2. Network Flows vs. TCP/IP Five-Tuple

### The Five-Tuple
A network flow aggregates packets belonging to the same communication session. This session is mathematically identified by the **Five-Tuple**:
1. **Source IP Address** (Layer 3 - Network)
2. **Destination IP Address** (Layer 3 - Network)
3. **Source Port** (Layer 4 - Transport)
4. **Destination Port** (Layer 4 - Transport)
5. **Protocol** (Layer 4 - Transport, e.g., TCP, UDP, ICMP)

### Directionality: Forward (Fwd) vs. Backward (Bwd)
* **Forward (Fwd)**: Represents traffic flowing from the **initiator (client)** of the connection to the **recipient (server)**.
* **Backward (Bwd)**: Represents the response traffic flowing from the **recipient (server)** back to the **initiator (client)**.

---

## 3. Network Anatomy of Attacks in CICIDS2017

To explain what happens during an attack to an examiner, memorize these direct network signatures:

```text
┌─────────────────┬────────────────────────────────────────────────────────┐
│ Attack Class    │ Network Behavior & Flag Alterations                    │
├─────────────────┼────────────────────────────────────────────────────────┤
│ DDoS / DoS      │ Flooding target ports with SYN, UDP, or HTTP requests  │
│                 │ to exhaust memory queues. Spikes TCP SYN/ACK flags,    │
│                 │ crushes Flow IAT (Inter-Arrival Times) to near zero.   │
├─────────────────┼────────────────────────────────────────────────────────┤
│ PortScan        │ Probing range of ports (1-65535) using quick SYN, TCP  │
│                 │ connect, or FIN scans. Creates massive forward flow    │
│                 │ volume with zero backward response bytes.              │
├─────────────────┼────────────────────────────────────────────────────────┤
│ Botnets         │ Infected machines check in periodically with a command │
│                 │ & control (C2) server. Identified by distinct, highly  │
│                 │ regular Active/Idle state timing cycles (beacons).      │
├─────────────────┼────────────────────────────────────────────────────────┤
│ Brute Force     │ Automated scripts making thousands of rapid login      │
│                 │ attempts (FTP/SSH). Spikes connection teardown rates   │
│                 │ (RST/FIN flags) on specific standard ports (21, 22).   │
└─────────────────┴────────────────────────────────────────────────────────┘
```

---

## 4. Comprehensive Column Reference & Attack Mappings

Here is a full breakdown of **every feature column** extracted by CICFlowMeter in your dataset.

### Category A: Connection Identification & Duration
#### 1. `Destination Port`
* **What it is**: The TCP or UDP port number on the recipient host (range: $0$ to $65535$).
* **Attack Relevance**: Core feature for separating normal traffic (e.g., HTTP on 80, HTTPS on 443) from attacks. **PortScans** target a wide range of sequential ports. **Brute Force** targets administrative ports (SSH on 22, FTP on 21). **DoS/DDoS** focuses on specific target ports to exhaust them.

#### 2. `Flow Duration`
* **What it is**: The total elapsed time of the communication session, from the first packet to the last packet (measured in microseconds).
* **Attack Relevance**: **DDoS floods** initiate connections and tear them down immediately, resulting in ultra-low flow durations. **Slowloris DoS** attacks hold connections open as long as possible, resulting in extremely high flow durations.

---

### Category B: Traffic Volume & Size Metrics
#### 3. `Total Fwd Packets` & 4. `Total Backward Packets`
* **What they are**: The total count of packets sent in the Forward (client to server) and Backward (server to client) directions.
* **Attack Relevance**: **High-volume floods (DDoS)** exhibit massive packet counts in the forward direction. **PortScans** send 1 or 2 forward packets and receive 0 or 1 backward packets.

#### 5. `Total Length of Fwd Packets` & 6. `Total Length of Bwd Packets`
* **What they are**: The cumulative size of all packet payloads (in bytes) sent in the Forward and Backward directions.
* **Attack Relevance**: **Data exfiltration** exhibits massive forward lengths. **DDoS floods** like GoldenEye send large HTTP request payloads, spiking forward lengths. In **Slowloris**, lengths are tiny because the attacker sends only incomplete headers.

#### 7. `Subflow Fwd Packets`, 8. `Subflow Fwd Bytes`, 9. `Subflow Bwd Packets`, 10. `Subflow Bwd Bytes`
* **What they are**: Subflows partition long-running connections into smaller statistical windows. These track the average packets and bytes sent in forward/backward sub-sessions.
* **Attack Relevance**: Used to detect long-running, stealthy attacks (like **Botnet C2 channels**) that try to hide large volumes by spreading them across long time periods.

#### 11. `act_data_pkt_fwd` (Actual Data Packets Forward)
* **What it is**: The count of forward packets containing at least 1 byte of actual TCP payload (excludes pure control packets like SYN, ACK, and RST).
* **Attack Relevance**: High values represent actual data transfer (normal traffic or Web Attacks). A value of $0$ is highly indicative of **SYN Floods** or **PortScans**, which send only control headers.

---

### Category C: Packet Size Distribution
#### 12. `Fwd Packet Length Max` / 13. `Min` / 14. `Mean` / 15. `Std`
#### 16. `Bwd Packet Length Max` / 17. `Min` / 18. `Mean` / 19. `Std`
#### 20. `Min Packet Length` / 21. `Max Packet Length`
#### 22. `Packet Length Mean` / 23. `Std` / 24. `Variance`
#### 25. `Average Packet Size`
#### 26. `Avg Fwd Segment Size` & 27. `Avg Bwd Segment Size`
* **What they are**: Statistical descriptors (minimum, maximum, average, standard deviation, and variance) of the sizes of packets (in bytes) sent in both directions. `Segment Size` measures the average size of TCP segments.
* **Attack Relevance**: 
  * Normal web browsing has highly variable packet sizes (large downloads, small requests), resulting in high **Packet Length Variance** and **Std**.
  * **DDoS SYN floods** and **PortScans** use highly standardized, identical packet sizes (typically 40 or 60 bytes of pure headers), resulting in **zero variance** and identical Min/Max lengths.
  * **Avg Bwd Segment Size** is a critical indicator: if a client sends a request and the server responds with zero data (like in a rejected port scan), Bwd Segment Size drops to $0$.

---

### Category D: Throughput Rate Metrics
#### 28. `Flow Bytes/s`
* **What it is**: The rate of data transfer (total bytes sent + received divided by flow duration).
* **Attack Relevance**: Spikes during large data transfers or high-volume **HTTP floods (Hulk/GoldenEye)**. Drops to near zero during **Slowloris DoS** attacks.

#### 29. `Flow Packets/s`, 30. `Fwd Packets/s`, 31. `Bwd Packets/s`
* **What they are**: The rate of packet transfer (total packets, forward packets, or backward packets divided by flow duration).
* **Attack Relevance**: **DDoS floods** produce massive spikes in `Fwd Packets/s`. **PortScans** yield high rates for short durations.

---

### Category E: Inter-Arrival Time (IAT) Timing Metrics
#### 32. `Flow IAT Mean` / 33. `Std` / 34. `Max` / 35. `Min` (Flow Inter-Arrival Time)
#### 36. `Fwd IAT Total` / 37. `Mean` / 38. `Std` / 39. `Max` / 40. `Min` (Forward IAT)
#### 41. `Bwd IAT Total` / 42. `Mean` / 43. `Std` / 44. `Max` / 45. `Min` (Backward IAT)
* **What they are**: Statistical measures of the time intervals (in microseconds) elapsed between consecutive packet arrivals. `Total` is the sum of all IATs.
* **Attack Relevance**: timing metrics are the most powerful indicators for distinguishing human traffic from automated attacks:
  * **Human Web Traffic**: Highly irregular, with bursts followed by long thinking periods, resulting in very high **Flow IAT Max** and **Std**.
  * **DDoS Floods**: Automated packet generators fire packets as fast as physically possible. **Flow IAT Mean** and **Min** drop to near $0$ microseconds.
  * **Botnet Beacons**: C2 check-ins have highly regular intervals, resulting in static, predictable **Flow IAT Mean** with almost **zero standard deviation**.

---

### Category F: Header Size & Structure
#### 46. `Fwd Header Length` & 47. `Bwd Header Length` (also `Fwd Header Length.1`)
* **What they are**: The total size (in bytes) of the IP and TCP/UDP headers sent in both directions.
* **Attack Relevance**: Spikes during **high-volume header floods (SYN Floods)**. Helps the model calculate the ratio of headers to payload: normal traffic has a high payload-to-header ratio, while attacks have a high header-to-payload ratio.

#### 48. `min_seg_size_forward`
* **What it is**: The minimum observed segment size in the forward direction, which indicates the minimum size of the TCP header configuration (usually 20 bytes).
* **Attack Relevance**: Custom attack tools and raw socket scripts often forge TCP headers, producing illegal or highly abnormal `min_seg_size_forward` values that deviate from standard OS TCP stack templates (e.g. less than 20 bytes), flagging them immediately.

---

### Category G: TCP Control Flags & State
TCP flags manage connection state (initiating, synchronizing, pushing, resetting, and tearing down connections). Abnormal flag combinations are the core signature of network intrusions.

```
TCP Handshake & Flags:
┌───────────┐         SYN Packet (SYN Flag = 1)          ┌───────────┐
│           ├───────────────────────────────────────────►│           │
│  Client   │      SYN-ACK Packet (SYN/ACK Flags = 1)    │  Server   │
│ (Fwd Flow)◄────────────────────────────────────────────┤(Bwd Flow) │
│           │         ACK Packet (ACK Flag = 1)          │           │
│           ├───────────────────────────────────────────►│           │
└───────────┘                                            └───────────┘
```

#### 49. `Fwd PSH Flags` & 50. `Bwd PSH Flags`
* **What they are**: The count of packets with the **PSH (Push)** flag set. PSH forces the receiver's TCP buffer to pass data directly to the application layer immediately rather than waiting for a full buffer.
* **Attack Relevance**: Spikes in interactive attacks (like **Web Attacks** or **Brute Force**) because the attacker needs immediate responses from the server.

#### 51. `Fwd URG Flags` & 52. `Bwd URG Flags`
* **What they are**: The count of packets with the **URG (Urgent)** flag set, indicating that specific data within the packet is high-priority and should be processed out-of-band.
* **Attack Relevance**: Rarely used in modern network traffic. Spikes are highly indicative of older denial-of-service attempts or custom packet crafted **probing attacks**.

#### 53. `FIN Flag Count`
* **What it is**: The count of packets with the **FIN (Finish)** flag set, requesting connection teardown.
* **Attack Relevance**: Spikes during graceful connection terminations. High rates occur in **Brute Force** scripts that continuously open and close connections rapidly.

#### 54. `SYN Flag Count`
* **What it is**: The count of packets with the **SYN (Synchronize)** flag set, which initiates the TCP three-way handshake.
* **Attack Relevance**: **SYN Floods (DDoS)** flood the server with SYN packets without completing the handshake. **PortScans (SYN Scans)** send SYN packets to check if a port is listening. Both cause massive spikes in `SYN Flag Count`.

#### 55. `RST Flag Count`
* **What it is**: The count of packets with the **RST (Reset)** flag set, which forcefully terminates a connection due to an error.
* **Attack Relevance**: Spikes during **PortScans** because when the scanner probes a closed port, the host operating system responds with an RST packet to close the connection.

#### 56. `PSH Flag Count` & 57. `ACK Flag Count`
* **What they are**: Overall count of PSH and ACK flags across all packets in the flow.
* **Attack Relevance**: **ACK Flag Count** spikes in normal established connections and **ACK floods (DDoS)**.

#### 58. `URG Flag Count`, 59. `CWE Flag Count`, 60. `ECE Flag Count`
* **What they are**: Counts of URG, CWE (Congestion Window Reduced), and ECE (ECN-Echo) flags. ECE and CWE are used for network congestion notifications.
* **Attack Relevance**: Custom crafted packets from reconnaissance scanners (like Nmap) use abnormal, illegal flag combinations (e.g., Xmas scans setting FIN, PSH, and URG simultaneously) to map operating system responses, causing spikes in these counts.

#### 61. `Down/Up Ratio`
* **What it is**: The ratio of download (backward) traffic to upload (forward) traffic.
* **Attack Relevance**: Normal browsing is asymmetric (high download/low upload, high ratio). **DDoS floods** and **PortScans** are extremely upload-heavy (low ratio).

---

### Category H: Bulk Rate & High-Volume Metrics
#### 62. `Fwd Avg Bytes/Bulk` / 63. `Fwd Avg Packets/Bulk` / 64. `Fwd Avg Bulk Rate`
#### 65. `Bwd Avg Bytes/Bulk` / 66. `Bwd Avg Packets/Bulk` / 67. `Bwd Avg Bulk Rate`
* **What they are**: Metrics that calculate data volume, packet count, and transmission rate during "bulk" transfers (periods where data is sent continuously without idle gaps).
* **Attack Relevance**: Spikes during massive data exfiltration or high-volume **DDoS HTTP floods**. Remains at 0 during slow, stealthy attacks (like Slowloris or Botnet beacons).

---

### Category I: Buffer & Window Allocation
#### 68. `Init_Win_bytes_forward`
* **What it is**: The number of bytes allocated to the initial TCP receiver window in the Forward direction (first packet). This tells the sender how much data the client can receive before requiring an ACK.
* **Attack Relevance**: Standard operating systems (Windows, Linux, macOS) assign predictable, fixed default window sizes (e.g., `8192` or `65535`). Custom attack tools (like those used in **DoS/DDoS**) often set arbitrary or raw values (e.g. `0` or random numbers) to bypass the local OS stack, making this one of the **highest-ranked SHAP features** for identifying synthesized attack traffic.

#### 69. `Init_Win_bytes_backward`
* **What it is**: The number of bytes allocated to the initial TCP receiver window in the Backward direction (server response).
* **Attack Relevance**: Similar to forward windows. If a client targets a closed port (like in a **PortScan**), the server cannot establish a window, causing `Init_Win_bytes_backward` to drop to `-1` or `0`, signaling an aborted connection attempt.

---

### Category J: Host State & Timing Cycles
#### 70. `Active Mean` / 71. `Active Std` / 72. `Active Max` / 73. `Active Min`
* **What they are**: Statistical measures of the time (in microseconds) a flow was active before going idle.
* **Attack Relevance**: Tracks the duration of active attack bursts.

#### 74. `Idle Mean` / 75. `Idle Std` / 76. `Idle Max` / 77. `Idle Min`
* **What they are**: Statistical measures of the time (in microseconds) a flow sat completely idle before transmitting data again.
* **Attack Relevance**: **Critical for Botnet detection**. Botnets sit completely idle for long periods (e.g., 300 seconds) between periodic C2 command polls, resulting in large, highly static, low-variance `Idle Mean` values.

---

  * `4`: DoS
  * `5`: PortScan
  * `6`: Web Attack

---

## 🎯 30 Crux Viva Questions & Answers (Topic 2)

### Q1: What is the difference between a raw packet (PCAP) and a network flow?
**A**: A raw packet contains every byte of the transmission including data payloads and individual transport headers. A network flow is a statistical summary of a connection session between two hosts. It aggregates packet details into metadata metrics (timing, averages, flag counts) associated with a single logical session.

### Q2: What elements make up the network TCP/IP Five-Tuple?
**A**: Source IP Address, Destination IP Address, Source Port, Destination Port, and the Transport Protocol (TCP, UDP, or ICMP).

### Q3: Explain Layer 1 (Physical Layer) in the context of this project.
**A**: Layer 1 deals with raw electromagnetic bitstreams over physical mediums (cables, fiber). In our NIDS, this is represented by physical network interface cards (NICs) connected to a Switch Port Analyzer (SPAN) or a physical hardware tap that mirrors and captures signals to be parsed.

### Q4: Why did your project preprocessor discard Layer 2 (Data Link Layer) MAC addresses?
**A**: MAC addresses are local node-to-node physical identifiers that change at every router hop. If a machine learning model learned MAC address rules, it would completely fail to generalize outside the specific testbed network it was trained on. Pruning them is mandatory for real-world generalization.

### Q5: At which layer does DDoS IP spoofing occur, and how does your NIDS analyze L3 traffic?
**A**: It occurs at Layer 3 (Network Layer) where IP packets are routed. Attackers spoof source IP addresses to evade standard IP firewalls. Our NIDS analyzes L3 metrics by evaluating aggregated statistical features like total bytes, packet sizes, and flow volumes rather than relying on static IP blacklists.

### Q6: Explain TCP's three-way handshake (Layer 4) and how a SYN Flood exploits it.
**A**: TCP handshakes require a client to send a `SYN` (Synchronize) packet, the server to respond with a `SYN-ACK` (Synchronize-Acknowledge), and the client to complete it with an `ACK` (Acknowledge) packet. A SYN Flood floods the server with SYN packets but never sends the final ACK. This leaves connection queues half-open, exhausting server memory and crashing the service.

### Q7: At which layer does a PortScan operate, and what is its goal?
**A**: Layer 4 (Transport Layer). Its goal is to systematically probe ports (1 to 65535) using TCP handshakes or raw UDP pings to identify open ports and map the services running on the victim host.

### Q8: Explain how a Slowloris attack exploits Layer 5 (Session Layer).
**A**: A Slowloris DoS attack opens thousands of active connection sockets (Layer 5 sessions) with a web server and transmits incomplete HTTP headers extremely slowly. Because the sessions remain open waiting for completion, the server's session pool is fully exhausted, blocking legitimate users from connecting.

### Q9: Why can standard Deep Packet Inspection (DPI) firewalls fail at Layer 6 (Presentation Layer)?
**A**: Layer 6 handles formatting and encryption (SSL/TLS). DPI firewalls rely on inspecting application layer payloads. If the traffic is encrypted using HTTPS at Layer 6, the firewall cannot read the payload without decrypting it, rendering payload-based defenses useless.

### Q10: How does a flow-based ML model bypass Layer 6 encryption?
**A**: Flow-based models bypass encryption by ignoring packet payloads entirely. Instead, they look at metadata timing shapes, packet size distributions, and flag states at L3/L4. Encrypted attacks still produce distinct high-frequency timing and size distributions that standard ML models classify with high precision.

### Q11: What types of attacks target Layer 7 (Application Layer) in this project?
**A**: Web Attacks (SQL Injection and Cross-Site Scripting targeting HTTP/HTTPS apps), Brute Force attacks (attempting unauthorized log-ins on FTP or SSH), and Botnet Command and Control (C2) channels.

### Q12: Why does a Brute Force login script target Layer 7 but get caught at Layer 4?
**A**: The Brute Force script targets the application authentication portal (Layer 7). However, to rapidly guess passwords, it initiates thousands of sequential connections. This automated speed produces extreme Layer 4 TCP signatures—specifically massive rates of `FIN Flag Count` and `RST Flag Count` as connections are rapidly opened and forcefully torn down.

### Q13: What is the fundamental difference between TCP and UDP at the Transport Layer?
**A**: TCP is connection-oriented, reliable, and uses handshakes, window controls, and flags. UDP is connectionless, unreliable, and fires datagrams without verifying delivery. In our dataset, UDP flows have `SYN/ACK/RST Flag Counts` strictly set to 0.

### Q14: How does `Flow Duration` distinguish a DDoS flood from a Slowloris attack?
**A**: DDoS floods use rapid, high-speed automated packet bursts that tear down immediately, resulting in ultra-low `Flow Duration` (near-zero microseconds). Slowloris DoS intentionally keeps connections open as long as possible, resulting in extremely high `Flow Duration` metrics.

### Q15: Why does `act_data_pkt_fwd` drop to $0$ during SYN floods or PortScans?
**A**: `act_data_pkt_fwd` counts packets containing at least one byte of application payload. SYN floods and PortScans send exclusively TCP control headers (pure SYN or RST packets) with zero application bytes, causing this metric to drop strictly to 0.

### Q16: How does `Packet Length Variance` differ between normal web browsing and a DDoS flood?
**A**: Normal browsing has highly variable packet sizes (large HTML/image downloads mixed with small ACK requests), resulting in high `Packet Length Variance`. DDoS floods utilize highly standardized, identical automated packet frames (e.g. 64-byte SYN packets), resulting in zero or near-zero variance.

### Q17: Explain why `Avg Bwd Segment Size` is an excellent indicator for identifying PortScans.
**A**: During a PortScan, the scanner targets closed ports. The victim host immediately rejects these connections by sending an `RST` packet (which has zero payload data). Consequently, the server sends zero bytes back, causing `Avg Bwd Segment Size` to drop to 0, which is a massive indicator of an aborted connection probe.

### Q18: Why are packet inter-arrival times (IATs) highly effective for separating human and automated traffic?
**A**: Humans click a link, wait for a page to load, and read (generating bursty traffic separated by massive, random thinking pauses). Automated attack scripts generate packet bursts with highly uniform, rapid intervals, which is instantly flagged by low IAT mean and IAT standard deviations.

### Q19: What does a `Flow IAT Std` of nearly zero microseconds signify during a flood?
**A**: It signifies that packets are arriving at completely uniform, microsecond-exact intervals. This is a dead giveaway of automated packet generators (DDoS) because physical human inputs and standard network latency jitter always produce high standard deviations.

### Q20: How is `min_seg_size_forward` useful for flagging custom crafted packets?
**A**: Standard operating systems configure TCP segments using standard default sizes (e.g., 20 or 32 bytes). Custom hacker scripts or raw socket libraries often assemble TCP packets manually with invalid, custom-sized headers, creating abnormal `min_seg_size_forward` values that deviate from the OS norm.

### Q21: What is the role of the PSH (Push) TCP flag, and when does it spike?
**A**: The PSH flag pushes data immediately to the application buffer rather than waiting for the TCP buffer to fill. It spikes during interactive attacks—like Web Attacks (SQLi/XSS) or SSH Brute Forcing—where the attacker requires instantaneous, sequential shell responses.

### Q22: What is the significance of the URG (Urgent) flag in modern intrusion detection?
**A**: The URG flag indicates that specific packet data has a high priority and must be processed out-of-order. It is virtually unused in modern normal traffic. Any spike in `URG Flag Count` is a high indicator of custom packet crafting or legacy DoS exploits.

### Q23: Why does the `RST (Reset) Flag Count` spike during a PortScan?
**A**: PortScans probe hundreds of random ports. When a scanner targets a closed port, the host operating system's kernel automatically responds with a TCP Reset (`RST`) packet to block the illegal connection, creating a spike in the flow's RST flag count.

### Q24: How does `SYN Flag Count` help identify a DDoS attack?
**A**: A TCP connection starts with a `SYN` packet. In a SYN flood (DDoS), the attacker floods the target with millions of SYN packets without completing the handshake. Thus, the SYN count spikes massively relative to other flags.

### Q25: Explain the role of the `Down/Up Ratio` in distinguishing user browsing from a DDoS flood.
**A**: Normal user browsing downloads heavy web pages and uploads tiny HTTP requests, yielding a high Down/Up ratio. DDoS floods and PortScans are upload-heavy (blasting packets *into* the server with minimal responses), yielding extremely low ratios.

### Q26: Gotcha Q: If an attacker changes the source IP and source port dynamically for every single packet, how does `CICFlowMeter` log this traffic, and does it break our flow model?
**A**: It fragments the traffic. `CICFlowMeter` groups packets into flows based on the five-tuple (Src IP, Dst IP, Src Port, Dst Port, Protocol). If the attacker randomizes the source IP and port for every packet, the flow engine cannot aggregate them. It logs every single packet as an independent, single-packet flow of duration $0$. This does not break our model; instead, it creates a massive spike in high-frequency, single-packet anomaly flows, which the model instantly flags.

### Q27: Gotcha Q: Why do UDP flows in our dataset have features like `SYN Flag Count` and `ACK Flag Count`? Aren't those strictly TCP concepts?
**A**: They are strictly TCP header flags. However, `CICFlowMeter` uses a single flat schema for all extracted network flows to maintain tabular consistency for machine learning models. For UDP and ICMP flows, these TCP flag columns are simply populated with constant values of `0`. These act as zero-variance columns for UDP slices, which our preprocessing pipeline automatically prunes during protocol-specific training.

### Q28: Gotcha Q: If our model relies heavily on `Init_Win_bytes_forward` to detect attacks, could an attacker bypass the model by setting their attack tool to use the default Windows window size of `8192`?
**A**: Yes, this is a classic **signature evasion** or fingerprint bypass. If an ML model over-indexes on static operating system parameters like `Init_Win_bytes_forward` because the training set attacks were launched from simple raw-socket libraries (which default to 0 or 1024), the attacker can easily bypass this by configuring their sockets to mimic standard Windows (`8192`) or Linux (`29200`) signatures. This highlights why models must balance static fingerprints with high-dimensional statistical metrics like IAT and packet variance.

### Q29: Gotcha Q: Why does a packet's `Flow IAT Mean` increase when network congestion is high, and how does this affect our model's classification accuracy?
**A**: Network congestion introduces stochastic queuing delays in router buffers, which shifts the packet Inter-Arrival Time (IAT) distribution to the right, inflating both the mean and standard deviation. A model trained on a clean, low-congestion lab network might suffer from **covariate shift** under high-congestion production environments, misclassifying normal latency-jittered benign flows as slow anomalous probes (like Slowloris).

### Q30: Gotcha Q: If `min_seg_size_forward` is defined as a TCP header length, how can its value be negative or extremely large (like 8000000) in the dataset?
**A**: It is a data-corruption artifact caused by parsing bugs in `CICFlowMeter`. When capturing fragmented IP packets or out-of-order frames, the parser occasionally miscalculates the offset between the IP and TCP headers, resulting in underflows (which cast unsigned variables to negative numbers) or garbage values. In real network engineering, a TCP header can never exceed 60 bytes. We resolve this by sanitizing and scaling these extreme artifacts during data preprocessing.
