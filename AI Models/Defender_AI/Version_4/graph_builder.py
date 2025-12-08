import torch
import dns.resolver
import requests
from torch_geometric.data import Data
from urllib.parse import urlparse

class UrlGraphBuilder:
    def __init__(self):
        pass

    def get_live_ip(self, domain):
        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = 2
            resolver.lifetime = 2
            result = resolver.resolve(domain, 'A')
            return result[0].to_text()
        except Exception: # <--- FIXED: Only catch code errors, not Ctrl+C
            return None

    def get_subdomains_crt(self, domain):
        subdomains = set()
        try:
            url = f"https://crt.sh/?q=%.{domain}&output=json"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                for entry in data[:5]:
                    subdomains.add(entry['name_value'])
        except Exception: # <--- FIXED: Only catch code errors, not Ctrl+C
            pass
        return list(subdomains)

    def extract_features(self, text, node_type):
        f1 = float(node_type)
        f2 = min(len(text), 100) / 100.0
        f3 = min(text.count('.'), 10) / 10.0
        keywords = ['login', 'secure', 'account', 'verify', 'bank', 'update']
        f4 = 1.0 if any(k in text.lower() for k in keywords) else 0.0
        return [f1, f2, f3, f4]

    def get_features(self, url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = "http://" + url
            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]
            if not domain: domain = url
        except Exception:
            domain = url

        # --- NODES ---
        node_features = []
        node_features.append(self.extract_features(url, 0))
        node_features.append(self.extract_features(domain, 1))

        has_ip = False
        ip = self.get_live_ip(domain) # Now if you Ctrl+C here, it will actually stop!
        if ip:
            node_features.append(self.extract_features(ip, 2))
            has_ip = True

        subdomains = self.get_subdomains_crt(domain) # Ctrl+C here will also work now
        subdomain_indices = []
        for sub in subdomains:
            node_features.append(self.extract_features(sub, 3))
            subdomain_indices.append(len(node_features) - 1)

        # --- EDGES ---
        src = []
        dst = []
        src.extend([0, 1]); dst.extend([1, 0])

        if has_ip:
            src.extend([1, 2]); dst.extend([2, 1])

        for sub_idx in subdomain_indices:
            src.extend([1, sub_idx]); dst.extend([sub_idx, 1])

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        return data

if __name__ == "__main__":
    builder = UrlGraphBuilder()
    print("Test Scan on google.com...")
    g = builder.get_features("google.com")
    print(f"Nodes Found: {g.num_nodes}")
