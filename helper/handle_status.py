from starlette.responses import JSONResponse
from fastapi import status 

def handle_401(msg: str | None, custom_message: str | None = None):
    return JSONResponse(
        content={
            'message': 'Response with status 401',
            'code': status.HTTP_401_UNAUTHORIZED,
            'data': {
                'error': f'No {msg} found' if custom_message is None else custom_message
            }
        },
        status_code=status.HTTP_401_UNAUTHORIZED
    )

def handle_404(msg: str | None, custom_message: str | None = None):
    return JSONResponse(
        content={
            'message': 'Response with status 404',
            'code': status.HTTP_404_NOT_FOUND,
            'data': {
                'error': f'No {msg} found' if custom_message is None else custom_message
            }
        },
        status_code=status.HTTP_404_NOT_FOUND
    )

def handle_500(e):
    item = e.__dict__
    return JSONResponse(
        content={
            'message': 'Database error',
            'code': status.HTTP_500_INTERNAL_SERVER_ERROR,
            'data': {
                'system_code': item['orig'].args[0],
                'error': item['orig'].args[1],
            }
        },
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )

def handle_422(msg: str | None, custom_message: str | None = None):
    return JSONResponse(
        content={
            'message': 'Response with status 422',
            'code': status.HTTP_422_UNPROCESSABLE_ENTITY,
            'data': {
                'error': f'No {msg} found' if custom_message is None else custom_message
            }
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )