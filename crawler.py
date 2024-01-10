import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pdfminer.high_level import extract_text
from pathlib import Path
import matplotlib.pyplot as plt

visited_urls = set()

def download_pdf(pdf_url, filename):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    return False

def count_files_in_directory(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def pdf_to_text(pdf_file):
    try:
        return extract_text(pdf_file)
    except Exception as e:
        print(f"Error converting PDF to text: {e}")
        return ""

def save_text(text, filename):
    if text.strip():  # Save only non-empty text
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    # Check if the domain is exactly the base domain or a subdomain of it
    is_domain_or_subdomain = parsed.netloc == base_domain
    has_en_path = '/en/' in parsed.path
    is_pdf = '.pdf' in parsed.path
    return is_domain_or_subdomain and has_en_path or is_pdf and bool(parsed.scheme)

def extract_and_save_text(soup, filename):
    text = soup.get_text(separator='\n', strip=True)
    save_text(text, filename)

def get_safe_filename(url):
    parsed_url = urlparse(url)
    # Include the entire domain and replace dots with underscores
    domain_formatted = parsed_url.netloc.replace('www.', '')
    # Remove leading and trailing slashes from the path and replace remaining slashes with underscores
    path_formatted = parsed_url.path.strip('/').replace('/', '_')
    # Combine the formatted domain and path
    return domain_formatted + '_' + path_formatted + '.txt'


def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    is_subdomain = any([parsed.netloc.endswith(base_domain), parsed.netloc == base_domain])
    has_en_path = '/en/' in parsed.path
    return is_subdomain and has_en_path and bool(parsed.scheme)

def is_image_url(url):
    lower_url = url.lower()
    return lower_url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'))


def crawl_and_extract(url, base_url, base_domain, folder, counters, stats):
    if url in visited_urls:
        return
    visited_urls.add(url)
    counters['total_visits'] += 1


    try:
        response = requests.get(url)
        if response.status_code != 200:
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        page_filename = os.path.join(folder, get_safe_filename(url))
        extract_and_save_text(soup, page_filename)
        counters['file_count'] = count_files_in_directory(folder)

        # Record the current state
        stats.append((counters['file_count'], counters['total_visits']))
        plot_stats(stats)

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if not is_image_url(full_url) and is_valid_url(full_url, base_domain):
                if full_url.endswith('.pdf'):
                    pdf_filename = os.path.join(folder, get_safe_filename(url) + "__" + os.path.basename(full_url))
                    if download_pdf(full_url, pdf_filename):
                        text = pdf_to_text(pdf_filename)
                        text_filename = pdf_filename.replace('.pdf', '.txt')
                        save_text(text, text_filename)
                        counters['file_count'] = count_files_in_directory(folder)
                        stats.append((counters['file_count'], counters['total_visits']))
                        plot_stats(stats)
                        
                else:
                    crawl_and_extract(full_url, base_url, base_domain, folder, counters, stats)
    except Exception as e:
        print(f"Error crawling {url}: {e}")

def plot_stats(stats):
    file_counts, visit_counts = zip(*stats)
    plt.plot(visit_counts, file_counts)
    plt.xlabel('Total Visits')
    plt.ylabel('Number of Files')
    plt.title('Files vs. Visits Over Time')
    plt.savefig("C:/Users/petro/koo.png")

def main():
    base_url = 'http://www.tuc.gr/en/home'
    base_domain = 'tuc.gr'
    start_url = base_url
    output_folder = 'tuc_db'
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    counters = {'file_count': 0, 'total_visits': 0}
    stats = []

    crawl_and_extract(start_url, base_url, base_domain, output_folder, counters, stats)
    plot_stats(stats)

if __name__ == "__main__":
    main()