from abc import ABC, abstractmethod
from pandas import Series

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class PreProcessor(ABC):
    @abstractmethod
    def preprocessCorpus( self, Ids: Series, Corpus: Series, Flags: str ) -> Series:
        pass

class PreProcessorFactory( ABC ):
    @abstractstatic
    def getInstance() -> PreProcessor:
        pass
