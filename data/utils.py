import boto3
import requests


def download_binary_file(file_url: str, output_path: str) -> None:
    """
    Download binary data file from a URL.

    Args:
    ----
        file_url: URL where the file is hosted.
        output_path: Output path for the downloaded file.

    Returns
    -------
        None.
    """
    request = requests.get(file_url)
    with open(output_path, "wb") as f:
        f.write(request.content)
    print(f"Downloaded data from {file_url} at {output_path}")


def download_from_s3(s3_url: str, save_path: str) -> None:
    """
    Downloads a h5ad file from s3 to a local path

    Parameters
    ----------
    s3_url : str
        s3 url to download from
    save_path : str
        local path to save to

    Returns
    -------
    None
    """
    # Check the file and hash it's url and modified date
    s3 = boto3.resource("s3")
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)
    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    s3_object.download_file(save_path)
