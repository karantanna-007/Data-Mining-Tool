CREATE DATABASE datamining_db;
CREATE USER 'datamining_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON datamining_db.* TO 'datamining_user'@'localhost';
FLUSH PRIVILEGES;