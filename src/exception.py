import sys

def error_message(error, error_details:sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line number: {line_number} with error message: {str(error)}"
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_details = error_details

    def __str__(self):
        return error_message(self.error_message, self.error_details)