import os
import logging
from datetime import datetime


log_file=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",log_file)
os.makedirs(os.path.dirname(log_path),exist_ok=True)


logging.basicConfig(
    filename=log_path,
    format='[%(asctime)s]-%(name)s-%(lineno)d-%(levelname)s-%(message)s',
    level=logging.INFO
)









