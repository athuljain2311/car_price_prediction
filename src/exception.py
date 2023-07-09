import sys
from src.logger import logging

def error_message_details(error,error_detail):
    _,_,tb = error_detail.exc_info() # TRACEBACK
    file_name = tb.tb_frame.f_code.co_filename
    message = "ERROR OCCURED IN PYTHON SCRIPT : [{0}], IN LINE NUMBER : [{1}], WITH ERROR MESSAGE : [{2}]".format(file_name,tb.tb_lineno,str(error))
    return message

class CustomException(Exception):
    def __init__(self,error,error_detail):
        self.message = error_message_details(error,error_detail)
    
    def __str__(self):
        return self.message
