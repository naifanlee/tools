import logging

def create_logger(level=logging.INFO):
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)


if __name__ == '__main__':
    create_logger()
    logging.info('test')