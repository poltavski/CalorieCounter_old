import io
import requests
from PIL import Image
import numpy as np
import argparse
import json
import os
from datetime import datetime


def arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path or url of the image")
    ap.add_argument("-ip", "--server_ip", required=False, default='0.0.0.0',
                    help="estimation server id address in format *.*.*.*")
    ap.add_argument("-port", "--server_port", required=False, default='5015',
                    help="estimation server port")
    ap.add_argument("-out", "--json_results_folder", required=False, default='json_results',
                    help="json results folder")
    return ap

def read_image(image_object):
    """
    Read image by path or URL or Array
    Args:
        image_object: image path on local machine or URL link or Array

    Returns:
        Image in Pillow format or None value if invalid input image object
    """
    if type(image_object) is np.ndarray:
        return Image.fromarray(image_object)

    if type(image_object) is str and len(image_object) == 0:
        return None

    if len(image_object) > 7 and \
            ('http://' == image_object[:7] or 'https://' == image_object[:8]):
        response = requests.get(image_object)
        image_bin = io.BytesIO(response.content)
    else:
        image_bin = open(image_object, "rb")

    image = Image.open(image_bin)

    return image


class EstimationTemplate(object):
    def __call__(self, image_object) -> dict:
        raise NotImplementedError()


class ServerEstimator(EstimationTemplate):
    def __init__(self,
                 server_ip: str = '0.0.0.0',
                 server_port: int = 5015):
        """
        Class constructor
        Args:
            server_ip: estimation server id address in format *.*.*.*
            server_port: estimation server port
        """
        self.address = 'http://{}:{}'.format(server_ip, server_port)

        print('sending request to: ', self.address)

        response = requests.get(
            url='{}/ping/'.format(self.address)
        )

        if response.status_code != 200:
            raise RuntimeError(
                'Can\'t connect to server: {}'.format(self.address)
            )

    def analyze_image(self, image: Image):
        """
        Server inference on image
        Args:
            image: image in Pillow format

        Returns:
            Dict with prediction in percents for food categories
        """
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='JPEG')
        img_byte_array = img_byte_array.getvalue()

        response = requests.post(
            url='{}/analyse/'.format(
                self.address
            ),
            files={"file": ("filename", img_byte_array, "image/jpeg")}
        )

        if response.status_code != 200:
            raise RuntimeError(
                'Request fail with code: {}'.format(response.status_code)
            )

        results = response.json()
        # print(results)
        return results

    def __call__(self, image_object) -> dict:
        """
        Insert image description to database
        Args:
            image_object: image path on local machine or URL link or Array

        Returns:
            Prediction in dictionary format
        """

        image = read_image(image_object)

        return self.analyze_image(image)
        # return self.analyze_image(image_object)


def main():
    args = arguments().parse_args()
    estimator = ServerEstimator(server_ip=args.server_ip, server_port=args.server_port)
    image_path = args.image
    json_results = estimator(image_object=image_path)
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

    os.makedirs(args.json_results_folder, exist_ok=True)
    json_result_path = os.path.join(args.json_results_folder, date_time+'.json')
    with open(json_result_path, 'w') as outfile:
        json.dump(json_results, outfile, indent=4)
        print('json results contained in: ', json_result_path)
        print('json results: ', json_results)


if __name__ == '__main__':
    main()

