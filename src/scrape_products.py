import ssl
import requests
from multiprocessing.pool import ThreadPool
import os
import time
import argparse
# import readline
# readline.parse_and_bind("tab: complete")


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
        img_data = requests.get(url, stream=True, timeout=120).content
        with open(file_out, 'wb') as handler:
            handler.write(img_data)
        print(f'>>> Fetched and saved image: {file_out}')
        return True
    except:
        return False


def main(dir_products):

    # classes_all = []

    # # in each folder, find urls.txt file and download URL in each line
    # for folder in sorted(os.listdir(dir_litw), key=str.casefold):
    #     if not os.path.isdir(os.path.join(dir_litw, folder)):
    #         continue
    #     classes_all.append(folder

    print(time.strftime("%H:%M:%S %Z"),
        f'>>> Images download started ...', end='\n')

    with open(os.path.join(dir_products, "products_url.csv"), 'r', errors='ignore') as txtfile:
        start = time.time()
        img_ids, urls = zip(*[line.split(',')[1:3]
                            for line in txtfile.readlines()])
        filepaths = [os.path.join(
            dir_products, 'products', f'img{img_id}.jpg') for img_id in img_ids]

        results = ThreadPool(20).imap_unordered(
            fetch_url, zip(urls, filepaths))
        results = list(results)

        end = time.time()
        print(f'>>> {sum(results)} images dowloaded in {end-start:.2f} sec!')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        '--path', type=str, default=os.path.abspath('C:/Users/melli/OneDrive/insight/data/'),
        help='Provide the path to the directory of the product data'
    )
    args = parser.parse_args()

    dir_litw = args.path

    main(dir_litw)
