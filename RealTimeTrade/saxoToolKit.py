from saxo_openapi.endpoints.rootservices.base import RootService
from saxo_openapi.endpoints.decorators import endpoint


def create_new_request(url):
    @endpoint(f"{url}")
    class OwnUrl(RootService):
        """Send a any request and get a 200 OK response with verb, url,
        headers and body in the response body.
        """
        RESPONSE_DATA = None

        def __init__(self):
            super(OwnUrl, self).__init__()

    REQUEST = OwnUrl()
    return REQUEST
