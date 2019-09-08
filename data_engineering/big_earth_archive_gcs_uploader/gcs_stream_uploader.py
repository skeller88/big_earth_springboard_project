from google.auth.transport.requests import AuthorizedSession
from google.resumable_media import requests, common
from google.cloud import storage


class GCSObjectStreamUploader(object):
    """
    From https://dev.to/sethmichaellarson/python-data-streaming-to-google-cloud-storage-with-resumable-uploads-458h
    """
    def __init__(self, client: storage.Client, bucket_name: str, blob_name: str, chunk_size: int = 256 * 1024):
        self._client = client
        self._bucket = self._client.bucket(bucket_name)
        self._blob = self._bucket.blob(blob_name)

        # stream (IO[bytes]): The stream (i.e. file-like object) to be uploaded during transmit_next_chunk
        self._stream = b''

        # total_bytes (Optional[int]): The (expected) total number of bytes in the ``stream``.
        self._total_bytes = 0

        # chunk_size (int): The size of the chunk to be read from the ``stream``
        self._chunk_size: int = chunk_size
        self._read = 0

        self._transport = AuthorizedSession(credentials=self._client._credentials)
        self._request: requests.ResumableUpload = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.stop()

    def start(self):
        url: str = f"https://www.googleapis.com/upload/storage/v1/b/{self._bucket.name}/o?uploadType=resumable"
        self._request = requests.ResumableUpload(
            upload_url=url, chunk_size=self._chunk_size
        )
        self._request.initiate(
            transport=self._transport,
            content_type='application/octet-stream',
            stream=self,
            stream_final=False,
            metadata={'name': self._blob.name},
        )

    def stop(self):
        self._request.transmit_next_chunk(self._transport)

    def write(self, data: bytes) -> int:
        data_len = len(data)
        self._total_bytes += data_len
        self._stream += data
        del data
        while self._total_bytes >= self._chunk_size:
            try:
                # Calls google.resumable_media._upload.get_next_chunk, which gets the next chunk using
                # self._stream, self._chunk_size, self._total_bytes.
                # See
                self._request.transmit_next_chunk(self._transport)
            except common.InvalidResponse:
                self._request.recover(self._transport)
        return data_len

    # with requests.get(url, stream=True) as r:
    #     shutil.copyfileobj(r.raw, local_fileobj)
    #     local_fileobj

    def read(self, chunk_size: int) -> bytes:
        # I'm not good with efficient no-copy buffering so if this is
        # wrong or there's a better way to do this let me know! :-)
        to_read = min(chunk_size, self._total_bytes)
        memview = memoryview(self._stream)
        self._stream = memview[to_read:].tobytes()
        self._read += to_read
        self._total_bytes -= to_read
        return memview[:to_read].tobytes()

    def tell(self) -> int:
        return self._read
