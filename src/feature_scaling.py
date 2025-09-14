"""
Feature scaling strategies for PySpark DataFrames.
Supports MinMaxScaler and StandardScaler transformations.
"""

import logging
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureScalingStrategy(ABC):
    """Abstract base class for feature scaling strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
        self.fitted_model = None
    
    @abstractmethod
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Scale specified columns in the DataFrame.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled features
        """
        pass


class ScalingType(str, Enum):
    """Enumeration of scaling types."""
    MINMAX = 'minmax'
    STANDARD = 'standard'


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """Min-Max scaling strategy to scale features to [0, 1] range."""
    
    def __init__(self, output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        """
        Initialize Min-Max scaling strategy.
        
        Args:
            output_col_suffix: Suffix to add to scaled column names
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
        logger.info("MinMaxScalingStrategy initialized (PySpark)")
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Min-Max scaling to specified columns.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        df_scaled = df 

        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            df_scaled = df_scaled.withColumn(col, F.col(col).cast("double"))
            assembler = VectorAssembler(inputCols=[col], outputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = MinMaxScaler(inputCol=vector_col, outputCol=scaled_vector_col)

            # pipeline = Pipeline(stages=[assembler, scaler])
            # pipeline_model = pipeline.fit(df_scaled)

            # get_value_udf = F.udf(lambda x: float(x[0] if x is not None else None), "double")
            # df_scaled = df_scaled.withColumn(
            #                                 col,
            #                                 get_value_udf(F.col(scaled_vector_col))
            #                                 )

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            # Create *_vec and *_scaled_vec
            df_scaled = pipeline_model.transform(df_scaled)

            # Replace original column with scalar from the 1-D vector
            df_scaled = df_scaled.withColumn(
                col,
                vector_to_array(F.col(scaled_vector_col)).getItem(0)
            )

            # Drop temp columns
            df_scaled = df_scaled.drop(vector_col, scaled_vector_col)

            # (Optional) keep the fitted scaler per column
            self.scaler_models[col] = pipeline_model.stages[-1]


        return df_scaled


class StandardScalingStrategy(FeatureScalingStrategy):
    """Standard scaling strategy to scale features to zero mean and unit variance."""
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, 
                 output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        """
        Initialize Standard scaling strategy.
        
        Args:
            with_mean: Whether to center the data before scaling
            with_std: Whether to scale the data to unit variance
            output_col_suffix: Suffix to add to scaled column names
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.with_mean = with_mean
        self.with_std = with_std
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
        logger.info(f"StandardScalingStrategy initialized (PySpark) - "
                   f"with_mean={with_mean}, with_std={with_std}")
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Standard scaling to specified columns.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        df_scaled = df 

        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            df_scaled = df_scaled.withColumn(col, F.col(col).cast("double"))

            assembler = VectorAssembler(inputCols=[col], outputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = StandardScaler(inputCol=vector_col, outputCol=scaled_vector_col)

            # pipeline = Pipeline(stages=[assembler, scaler])
            # pipeline_model = pipeline.fit(df_scaled)

            # get_value_udf = F.udf(lambda x: float(x[0] if x is not None else None), "double")
            # df_scaled = df_scaled.withColumn(
            #                                 col,
            #                                 get_value_udf(F.col(scaled_vector_col))
            #                                 )

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            df_scaled = pipeline_model.transform(df_scaled)

            df_scaled = df_scaled.withColumn(
                col,
                vector_to_array(F.col(scaled_vector_col)).getItem(0)
            )

            df_scaled = df_scaled.drop(vector_col, scaled_vector_col)
            self.scaler_models[col] = pipeline_model.stages[-1]

        return df_scaled