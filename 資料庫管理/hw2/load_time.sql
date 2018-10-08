LOAD DATA LOCAL INFILE '/Users/Apple/desktop/大三上/資料庫管理/hw2/時段'
REPLACE INTO TABLE mydb.Time_Session
FIELDS TERMINATED BY '\t'
lines TERMINATED BY '\n'