from typing import TypeVar
from biomed.utils.service_locator import ServiceLocator
from biomed.properties_manager import PropertiesManager
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.normalizer.simpleNormalizer import SimpleNormalizer
from biomed.preprocessor.facilitymanager.mFacilityManager import MariosFacilityManager
from biomed.preprocessor.cache.sharedMemoryCache import SharedMemoryCache
import biomed.preprocessor.cache.numpyArrayFileCache as NPC
import biomed.preprocessor.polymorph_preprocessor  as PP
import biomed.vectorizer.selector.selector_manager as SM
import biomed.vectorizer.std_vectorizer as Vect
import biomed.mlp.mlp_manager as MLP
from biomed.utils.simple_file_writer import SimpleFileWriter
from biomed.utils.json_file_writer import JSONFileWriter
from biomed.utils.csv_file_writer import CSVFileWriter
import biomed.evaluator.std_evaluator as Eval

__Services = ServiceLocator()
def startServices() -> None:
    #independent services
    __Services.set(
        "preprocessor.normalizer.simple",
        SimpleNormalizer.Factory()
    )

    __Services.set(
        "preprocessor.normalizer.complex",
        ComplexNormalizer.Factory()
    )

    __Services.set(
        "properties",
        PropertiesManager()
    )

    __Services.set(
        "preprocessor.facilitymanager",
        MariosFacilityManager.Factory.getInstance()
    )

    __Services.set(
        "preprocessor.cache.shared",
        SharedMemoryCache.Factory.getInstance()
    )

    __Services.set(
        "evaluator.simple",
        SimpleFileWriter.Factory.getInstance()
    )

    __Services.set(
        "evaluator.json",
        JSONFileWriter.Factory.getInstance()
    )

    __Services.set(
        "evaluator.csv",
        CSVFileWriter.Factory.getInstance()
    )

    #dependend services
    __Services.set(
        "preprocessor.cache.persistent",
        NPC.NumpyArrayFileCache.Factory.getInstance(),
        Dependencies = "properties"
    )

    __Services.set(
        "preprocessor",
        PP.PolymorphPreprocessor.Factory.getInstance(),
        Dependencies = [
            "preprocessor.facilitymanager",
            "preprocessor.normalizer.simple",
            "preprocessor.normalizer.complex",
            "preprocessor.cache.persistent",
            "preprocessor.cache.shared"
        ]
    )

    __Services.set(
        "vectorizer.selector",
        SM.SelectorManager.Factory.getInstance(),
        Dependencies = "properties"
    )

    __Services.set(
        "vectorizer",
        Vect.StdVectorizer.Factory.getInstance(),
        Dependencies = [
            "properties",
            "vectorizer.selector"
        ]
    )

    __Services.set(
        "mlp",
        MLP.MLPManager.Factory(),
        Dependencies = "properties"
    )

    __Services.set(
        "evaluator",
        Eval.StdEvaluator.Factory.getInstance(),
        Dependencies = [
            "properties",
            "evaluator.simple",
            "evaluator.json",
            "evaluator.csv"
        ]
    )

T = TypeVar( 'T' )
def getService( Key: str, ExpectedType ) -> T:
    return __Services.get( Key, ExpectedType )
