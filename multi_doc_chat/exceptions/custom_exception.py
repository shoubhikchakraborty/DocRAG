import sys
import traceback
from typing import Optional, Any

class DocumentPortalException(Exception):
    """
    This class gives:
    i) Custom error message
    ii) contextual info where an error happened
    iii) Full tracebacks
    iv) proper string output
    """

    def __init__(self, error_message: Any, error_details: Optional[Any] = None):
        # Normalize error message
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = error_message

        # Resolve traceback info
        exc_type = exc_value = exc_tb = None

        # Case 1: no details supplied -> use current exception context
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            # Case 2: explicit sys.exc_info() style tuple supplied
            if (
                isinstance(error_details, (tuple, list))
                and len(error_details) == 3
            ):
                exc_type, exc_value, exc_tb = error_details
            # Case 3: an exception instance was supplied
            elif isinstance(error_details, BaseException):
                exc_type = type(error_details)
                exc_value = error_details
                exc_tb = getattr(error_details, "__traceback__", None)
            # Case 4: object that exposes an exc_info() method (callable)
            elif callable(getattr(error_details, "exc_info", None)):
                try:
                    exc_type, exc_value, exc_tb = error_details.exc_info()
                except Exception:
                    # fallback
                    exc_type, exc_value, exc_tb = sys.exc_info()
            else:
                # final fallback
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Find the last traceback frame
        last_tb = exc_tb
        while last_tb and getattr(last_tb, "tb_next", None):
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Record traceback as string
        if exc_type and exc_tb:
            self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        base = f"Error in [{self.file_name}] at line [{self.lineno}] || Message: [{self.error_message}]"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"
