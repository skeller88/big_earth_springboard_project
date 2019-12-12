from pyspark.sql import *
from pyspark.sql.functions import *
import sparkdl

spark = SparkSession.builder \
    .appName("big_earth") \
    .getOrCreate()

df = spark.read.load('gs://big_earth/raw/json', format='json').withColumn("filename", input_file_name())
image_prefix_regex = r'gs:\/\/big_earth\/raw\/json\/(.*)_labels_metadata\.json'
df = df.withColumn('image_prefix', regexp_extract('filename', image_prefix_regex, 1))
df = df.withColumn('label_hash', sha2(array_join(array_sort(df.labels), '-'), 256))
df.write.mode('overwrite').parquet("gs://big_earth/metadata")
print(df.collect()[0])