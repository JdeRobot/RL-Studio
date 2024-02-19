import logging


class Logger:
    def __init__(self, log_file):
        # Configurar el formato de los registros
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.DEBUG, format=format)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(
            logging.WARNING
        )  # Solo se escribirán advertencias y errores en el archivo
        file_handler.setFormatter(logging.Formatter(format))
        logging.getLogger().addHandler(file_handler)

        # Configurar un manipulador para imprimir en la consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            logging.INFO
        )  # Solo se imprimirán mensajes de info y debug en la consola
        console_handler.setFormatter(logging.Formatter(format))
        logging.getLogger().addHandler(console_handler)

    def _info(self, message):
        logging.info(message)

    def _warning(self, message):
        logging.warning(message)

    def _error(self, message):
        logging.error(message)

    def _debug(self, message):
        logging.debug(message)


class LoggerAllInOne:
    def __init__(self, log_file):
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format=format)
        self.logger = logging.getLogger(__name__)

    def _info(self, message):
        self.logger.info(message)

    def _warning(self, message):
        self.logger.warning(message)

    def _error(self, message):
        self.logger.error(message)

    def _debug(self, message):
        self.logger.debug(message)


class LoggingHandler:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter(
            "[%(levelname)s] - %(asctime)s, filename: %(filename)s, funcname: %(funcName)s, line: %(lineno)s\n messages ---->\n %(message)s"
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
