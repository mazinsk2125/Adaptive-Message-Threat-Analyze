mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

# ✅ Download NLTK data to a location available at runtime
mkdir -p ~/.nltk_data
python -m nltk.downloader -d ~/.nltk_data stopwords punkt
