import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, from_unixtime

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = '{}song_data/*/*/*/*.json'.format(input_data)

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write \
        .partitionBy('year', 'artist_id') \
        .mode('overwrite') \
        .parquet('{}songs_table/'.format(output_data))

    # extract columns to create artists table
    artists_table = df \
        .select('artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude') \
        .dropDuplicates()

    # write artists table to parquet files
    artists_table.write \
        .mode('overwrite') \
        .parquet('{}artists_table/'.format(output_data))


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = "{}log-data/".format(input_data)

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table
    users_table = df.select(['userId', 'firstName', 'lastName', 'gender', 'level']).dropDuplicates()

    # write users table to parquet files
    users_table.write \
        .mode('overwrite') \
        .parquet('{}users_table'.format(output_data))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.timestamp(datetime.fromtimestamp(x/1000)))
    df = df.withColumn('ts_', get_timestamp(df.ts))

    # create datetime column from original timestamp column
    df = df.withColumn('datetime', from_unixtime(df.ts/1000))

    # extract columns to create time table
    time_table_cols = [df.ts_, hour(df.datetime).alias('hour'), dayofmonth(df.datetime).alias('day'),
                       weekofyear(df.datetime).alias('week'), month(df.datetime).alias('month'),
                       year(df.datetime).alias('year'), date_format(df.datetime, 'u').alias('weekday')]
    time_table = df.select(time_table_cols).dropDuplicates()

    # write time table to parquet files partitioned by year and month
    time_table.write \
        .partitionBy('year', 'month') \
        .mode('overwrite') \
        .parquet('{}time_table'.format(output_data))

    # read in song data to use for songplays table
    song_data = '{}song_data/*/*/*/*.json'.format(input_data)
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table
    songplays_songs_df = df.join(song_df, (df.song == song_df.title) & (df.artist == song_df.artist_name), how='left')
    songplays_table = songplays_songs_df.select(
        'ts_', 'userId', 'level', 'song_id', 'artist_id', 'sessionId', 'location', 'userAgent').dropDuplicates()

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write \
        .mode('overwrite') \
        .parquet('{}songplays_table'.format(output_data))


def main():
    spark = create_spark_session()
    input_data = "s3a://{}/".format(config['AWS']['S3_FOLDER'])
    output_data = "s3a://{}/".format(config['AWS']['S3_FOLDER'])

    # process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
