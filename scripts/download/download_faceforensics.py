#!/usr/bin/env python

""" Downloads FaceForensics++ and Deep Fake Detection public data release

Example usage:

    see -h or https://github.com/ondyari/FaceForensics

"""

# -*- coding: utf-8 -*-

import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join
import socket





# URLs and filenames

FILELIST_URL = 'misc/filelist.json'

DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'

DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5',]



# Parameters

DATASETS = {

    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',

    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',

    'original': 'original_sequences/youtube',

    'DeepFakeDetection_original': 'original_sequences/actors',

    'Deepfakes': 'manipulated_sequences/Deepfakes',

    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',

    'Face2Face': 'manipulated_sequences/Face2Face',

    'FaceShifter': 'manipulated_sequences/FaceShifter',

    'FaceSwap': 'manipulated_sequences/FaceSwap',

    'NeuralTextures': 'manipulated_sequences/NeuralTextures'

    }

ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',

                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',

                'NeuralTextures']

COMPRESSION = ['raw', 'c23', 'c40']

TYPE = ['videos', 'masks', 'models']

SERVERS = ['EU', 'EU2', 'CA']





def parse_args():

    parser = argparse.ArgumentParser(

        description='Downloads FaceForensics v2 public data release.',

        formatter_class=argparse.ArgumentDefaultsHelpFormatter

    )

    parser.add_argument('output_path', type=str, help='Output directory.')

    parser.add_argument('-d', '--dataset', type=str, default='all',

                        help='Which dataset to download, either pristine or '

                             'manipulated data or the downloaded youtube '

                             'videos.',

                        choices=list(DATASETS.keys()) + ['all']

                        )

    parser.add_argument('-c', '--compression', type=str, default='raw',

                        help='Which compression degree. All videos '

                             'have been generated with h264 with a varying '

                             'codec. Raw (c0) videos are lossless compressed.',

                        choices=COMPRESSION

                        )

    parser.add_argument('-t', '--type', type=str, default='videos',

                        help='Which file type, i.e. videos, masks, for our '

                             'manipulation methods, models, for Deepfakes.',

                        choices=TYPE

                        )

    parser.add_argument('-n', '--num_videos', type=int, default=None,

                        help='Select a number of videos number to '

                             "download if you don't want to download the full"

                             ' dataset.')

    parser.add_argument('--server', type=str, default='EU2',

                        help='Server to download the data from. Defaults to EU2 (kaldir server).',

                        choices=SERVERS

                        )

    parser.add_argument('-y', '--yes', action='store_true',
                        help='Automatically agree to TOS and run non-interactively.')

    args = parser.parse_args()



    # URLs

    server = args.server

    if server == 'EU':

        server_url = 'http://canis.vc.in.tum.de:8100/'

    elif server == 'EU2':

        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'

    elif server == 'CA':

        server_url = 'http://falas.cmpt.sfu.ca:8100/'

    else:

        raise Exception('Wrong server name. Choices: {}'.format(str(SERVERS)))

    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = server_url + 'v3/manipulated_sequences/Deepfakes/models/'

    return args





def download_files(filenames, base_url, output_path, report_progress=True, timeout=30):

    os.makedirs(output_path, exist_ok=True)

    if report_progress:
        filenames = tqdm(filenames)

    for filename in filenames:
        url = base_url + filename
        out_file = join(output_path, filename)
        download_file(url, out_file, report_progress=report_progress, timeout=timeout)





def reporthook(count, block_size, total_size):

    global start_time

    if count == 0:

        start_time = time.time()

        return

    duration = time.time() - start_time

    progress_size = int(count * block_size)

    speed = int(progress_size / (1024 * duration))

    percent = int(count * block_size * 100 / total_size)

    sys.stdout.write("\rProgress: %d%%, %d MB, %d KB/s, %d seconds passed" %

                     (percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()





def download_file(url, out_file, report_progress=False, max_retries=3, timeout=30):

    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(out_file):
        tqdm.write('WARNING: skipping download of existing file ' + out_file)
        return

    # Build fallback candidate URLs in case primary server is down
    candidate_urls = [url]
    known_servers = [
        'http://kaldir.vc.in.tum.de/faceforensics/v3/',
        'http://canis.vc.in.tum.de:8100/v3/',
        'http://falas.cmpt.sfu.ca:8100/v3/'
    ]
    for s_base in known_servers:
        if '/v3/' in url:
            rel_subpath = url.split('/v3/', 1)[1]
            alt_url = s_base + rel_subpath
            if alt_url not in candidate_urls:
                candidate_urls.append(alt_url)

    last_error = None
    for attempt in range(max_retries):
        for candidate_url in candidate_urls:
            try:
                fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
                f = os.fdopen(fh, 'w')
                f.close()

                req = urllib.request.Request(candidate_url)
                if report_progress:
                    urllib.request.urlretrieve(candidate_url, out_file_tmp, reporthook=reporthook)
                else:
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        with open(out_file_tmp, 'wb') as f:
                            f.write(response.read())

                os.rename(out_file_tmp, out_file)
                return  # Success
            except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
                last_error = e
                if os.path.exists(out_file_tmp):
                    try:
                        os.remove(out_file_tmp)
                    except Exception:
                        pass
                continue
            except Exception as e:
                if os.path.exists(out_file_tmp):
                    try:
                        os.remove(out_file_tmp)
                    except Exception:
                        pass
                raise

        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5
            tqdm.write(f'Retry {attempt + 1}/{max_retries} across servers after {wait_time}s: {str(last_error)}')
            time.sleep(wait_time)

    raise Exception(f'Failed to download {url} after trying all fallback servers: {str(last_error)}')





def main(args):

    # TOS

    print('By pressing any key to continue you confirm that you have agreed '\

          'to the FaceForensics terms of use as described at:')

    print(args.tos_url)

    print('***')

    if getattr(args, 'yes', False) or not sys.stdin.isatty() or os.environ.get('NONINTERACTIVE') == '1':
        print('Automated non-interactive mode: TOS agreed.')
    else:
        print('Press any key to continue, or CTRL-C to exit.')
        try:
            _ = input('')
        except (EOFError, KeyboardInterrupt):
            print('Non-interactive environment detected: TOS agreed.')



    # Extract arguments

    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS

    c_type = args.type

    c_compression = args.compression

    num_videos = args.num_videos

    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)



    # Check for special dataset cases

    for dataset in c_datasets:

        dataset_path = DATASETS[dataset]

        # Special cases

        if 'original_youtube_videos' in dataset:

            # Here we download the original youtube videos zip file

            print('Downloading original youtube videos.')

            if not 'info' in dataset_path:
                print('Please be patient, this may take a while (~40gb)')
                suffix = ''
            else:
                suffix = 'info'

            download_file(args.base_url + '/' + dataset_path,

                          out_file=join(output_path,

                                        'downloaded_videos{}.zip'.format(

                                            suffix)),

                          report_progress=True)

            return



        # Else: regular datasets

        print('Downloading {} of dataset "{}"'.format(

            c_type, dataset_path

        ))



        # Get filelists and video lenghts list from server
        # Try multiple servers if one fails
        servers_to_try = [args.base_url]
        if args.server == 'EU':
            servers_to_try.extend([
                'http://kaldir.vc.in.tum.de/faceforensics/v3/',
                'http://falas.cmpt.sfu.ca:8100/v3/'
            ])
        elif args.server == 'EU2':
            servers_to_try.extend([
                'http://canis.vc.in.tum.de:8100/v3/',
                'http://falas.cmpt.sfu.ca:8100/v3/'
            ])
        else:  # CA
            servers_to_try.extend([
                'http://canis.vc.in.tum.de:8100/v3/',
                'http://kaldir.vc.in.tum.de/faceforensics/v3/'
            ])

        filelist = None
        last_error = None
        
        for base_url_attempt in servers_to_try:
            try:
                if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
                    url = base_url_attempt + '/' + DEEPFEAKES_DETECTION_URL
                    response = urllib.request.urlopen(url, timeout=30)
                    filepaths = json.loads(response.read().decode("utf-8"))
                    if 'actors' in dataset_path:
                        filelist = filepaths['actors']
                    else:
                        filelist = filepaths['DeepFakesDetection']
                    args.base_url = base_url_attempt
                    print(f'Successfully connected to {base_url_attempt}')
                    break  # Success
                    
                elif 'original' in dataset_path:
                    url = base_url_attempt + '/' + FILELIST_URL
                    response = urllib.request.urlopen(url, timeout=30)
                    file_pairs = json.loads(response.read().decode("utf-8"))
                    filelist = []
                    for pair in file_pairs:
                        filelist += pair
                    args.base_url = base_url_attempt
                    print(f'Successfully connected to {base_url_attempt}')
                    break  # Success
                    
                else:
                    url = base_url_attempt + '/' + FILELIST_URL
                    response = urllib.request.urlopen(url, timeout=30)
                    file_pairs = json.loads(response.read().decode("utf-8"))
                    filelist = []
                    for pair in file_pairs:
                        filelist.append('_'.join(pair))
                        if c_type != 'models':
                            filelist.append('_'.join(pair[::-1]))
                    args.base_url = base_url_attempt
                    print(f'Successfully connected to {base_url_attempt}')
                    break  # Success
                    
            except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
                last_error = e
                print(f'Failed to connect to {base_url_attempt}: {str(e)}')
                if base_url_attempt != servers_to_try[-1]:
                    print(f'Trying next server...')
                    continue
                else:
                    print(f'All servers failed. Last error: {str(e)}')
                    raise Exception(f'Could not connect to any FaceForensics++ server. '
                                  f'Please check your internet connection or try again later. '
                                  f'Last error: {str(e)}')
        
        if filelist is None:
            raise Exception('Failed to retrieve file list from all servers')

        # Maybe limit number of videos for download
        if num_videos is not None and num_videos > 0:
            print('Downloading the first {} videos'.format(num_videos))
            filelist = filelist[:num_videos]



        # Server and local paths

        dataset_videos_url = args.base_url + '{}/{}/{}/'.format(

            dataset_path, c_compression, c_type)

        dataset_mask_url = args.base_url + '{}/{}/videos/'.format(

            dataset_path, 'masks', c_type)



        if c_type == 'videos':

            dataset_output_path = join(output_path, dataset_path, c_compression,

                                       c_type)

            print('Output path: {}'.format(dataset_output_path))

            filelist = [filename + '.mp4' for filename in filelist]

            download_files(filelist, dataset_videos_url, dataset_output_path)

        elif c_type == 'masks':

            dataset_output_path = join(output_path, dataset_path, c_type,

                                       'videos')

            print('Output path: {}'.format(dataset_output_path))

            if 'original' in dataset:

                if args.dataset != 'all':

                    print('Only videos available for original data. Aborting.')

                    return

                else:

                    print('Only videos available for original data. '

                          'Skipping original.\n')

                    continue

            if 'FaceShifter' in dataset:

                print('Masks not available for FaceShifter. Aborting.')

                return

            filelist = [filename + '.mp4' for filename in filelist]

            download_files(filelist, dataset_mask_url, dataset_output_path)



        # Else: models for deepfakes

        else:

            if dataset != 'Deepfakes' and c_type == 'models':

                print('Models only available for Deepfakes. Aborting')

                return

            dataset_output_path = join(output_path, dataset_path, c_type)

            print('Output path: {}'.format(dataset_output_path))



            # Get Deepfakes models

            for folder in tqdm(filelist):

                folder_filelist = DEEPFAKES_MODEL_NAMES



                # Folder paths

                folder_base_url = args.deepfakes_model_url + folder + '/'

                folder_dataset_output_path = join(dataset_output_path,

                                                  folder)

                download_files(folder_filelist, folder_base_url,

                               folder_dataset_output_path,

                               report_progress=False)   # already done





if __name__ == "__main__":

    args = parse_args()

    main(args)

