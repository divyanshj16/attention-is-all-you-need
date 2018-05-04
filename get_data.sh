# wget https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz -P ./dataset/
tar -xvzf dataset/de-en.tgz -C ./dataset/
python process_data.py
# rm ./dataset/de-en.tgz