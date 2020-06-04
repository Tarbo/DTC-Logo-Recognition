import ssl
import requests
from multiprocessing.pool import ThreadPool
import os
import time
import argparse
#import readline
#readline.parse_and_bind("tab: complete")

def fetch_url(url_file):
    """
    Fetch url and download content to fileself.

    Args:
      url_file: (url, file) tuple. If file already exists, mark as downloaded and skip.
    Returns:
      bool: True if download successfull, False otherwise.
    """
    url, file_out = url_file
    if os.path.exists(file_out):
        return True
    try:
        img_data = requests.get(url, stream=True, timeout=80).content
        with open(file_out, 'wb') as handler:
            handler.write(img_data)
        return True
    except:
        return False

def main(dir_litw):

    classes_all = []

    # in each folder, find urls.txt file and download URL in each line
    for folder in sorted(os.listdir(dir_litw), key=str.casefold):
        if not os.path.isdir(os.path.join(dir_litw, folder)):
            continue
        classes_all.append(folder)

        # no annotations in folder 0samples/
        if folder == '0samples':
            continue

        print(time.strftime("%H:%M:%S %Z"),
              f'>>> Downloading images in {folder} folder...',end='')

        with open(os.path.join(dir_litw, folder, 'urls.txt'), 'r', errors='ignore') as txtfile:
            start = time.time()

            img_ids, urls = zip(*[line.split('\t')
                                  for line in txtfile.readlines()])
            filepaths = [os.path.join(
                dir_litw, folder, f'img{img_id}.jpg') for img_id in img_ids]

            results = ThreadPool(20).imap_unordered(
                fetch_url, zip(urls, filepaths))
            results = list(results)

            end = time.time()
            print(f'>>> {sum(results)} images in {end-start:.2f} sec!')
    return classes_all


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        '--path', type=str, default=os.path.abspath('data/LogosInTheWild-v2/data'),
        help='path to Logos In The Wild data/ parent folder. Each subfolder contains a url.txt with links to images'
    )
    args = parser.parse_args()

    dir_litw = args.path

    main(dir_litw)
