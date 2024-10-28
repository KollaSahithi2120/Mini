import streamlit as st 
import sqlite3
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer 

# Custom CSS for font and size
st.markdown("""
    <style>
    /* Global font settings */
    body {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
    }

    /* Increase font size for the tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 20px;
        font-weight: bold;
    }

    /* Increase font size for titles */
    h1 {
        font-size: 36px !important;
    }

    h2 {
        font-size: 28px !important;
    }

    h3 {
        font-size: 24px !important;
    }

    /* Style for book details */
    .book-details {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('books.db')
    return conn

# Fetch books from the database
def fetch_books(search_query=""):
    conn = get_db_connection()
    query = "SELECT * FROM books WHERE book_name LIKE ? OR author_name LIKE ? OR genre LIKE ?"
    books = pd.read_sql(query, conn, params=[f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
    conn.close()
    return books

# Add a book to a list (wishlist or cart)
def add_to_list(book_id, list_type):
    if list_type not in st.session_state:
        st.session_state[list_type] = []
    if book_id not in st.session_state[list_type]:
        st.session_state[list_type].append(book_id)

# Display books with heart and cart buttons
def display_books(books):
    for i in range(0, len(books), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(books):
                row = books.iloc[i + j]
                with cols[j]:
                    st.image(row['image_url'], width=150)
                    st.write(f"<div class='book-details'><b>{row['book_name']}</b></div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><i>{row['author_name']}</i></div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><b>Genre:</b> {row['genre']}</div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><b>Published:</b> {row['yop']}</div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><b>Publisher:</b> {row['name_of_publisher']}</div>", unsafe_allow_html=True)

                    # Heart for Wishlist
                    if st.button(f"d Add to Wishlist", key=f"wishlist_{row['id']}"):
                        add_to_list(row['id'], 'wishlist')
                        st.success("Added to Wishlist")

                    # Button for Cart
                    if st.button("=� Add to Cart", key=f"cart_{row['id']}"):
                        add_to_list(row['id'], 'cart')
                        st.success("Added to Cart")

                    # View Details Button
                    if st.button("View Details", key=f"view_{row['id']}"):
                        st.write(row['description'])
            st.write("---")

# View books in a list (wishlist or cart)
def view_list(list_type):
    if list_type in st.session_state and st.session_state[list_type]:
        books = fetch_books()
        list_books = books[books['id'].isin(st.session_state[list_type])]
        display_books(list_books)
    else:
        st.write("Your list is empty.")

# Recommend books using TF-IDF and cosine similarity
def recommend_books():
    if 'wishlist' in st.session_state and st.session_state['wishlist']:
        books = fetch_books()
        wishlist_books = books[books['id'].isin(st.session_state['wishlist'])]

        # Combine the dataset descriptions (excluding wishlist books)
        dataset_books = books[~books['id'].isin(st.session_state['wishlist'])]
        all_descriptions = wishlist_books['description'].tolist() + dataset_books['description'].tolist()

        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_descriptions)

        # Calculate cosine similarity for wishlist books against the dataset
        wishlist_tfidf = tfidf_matrix[:len(wishlist_books)]
        dataset_tfidf = tfidf_matrix[len(wishlist_books):]

        similarity_scores = cosine_similarity(wishlist_tfidf, dataset_tfidf)

        # Get top recommendations excluding wishlist books
        top_k = 5
        recommendations = []

        for i, scores in enumerate(similarity_scores):
            top_indices = scores.argsort()[-top_k:][::-1]
            recommended_books = dataset_books.iloc[top_indices]
            recommendations.append({
                'wishlist_book': wishlist_books.iloc[i]['book_name'],
                'recommended_books': recommended_books[['book_name', 'image_url']].values
            })

        if recommendations:
            st.write("Based on your wishlist, we recommend the following books:")
            for rec in recommendations:
                st.write(f"*Wishlist Book:* {rec['wishlist_book']}")

                # Display recommended books in columns
                rec_cols = st.columns(len(rec['recommended_books']))
                for idx, (book_name, image_url) in enumerate(rec['recommended_books']):
                    with rec_cols[idx]:
                        st.image(image_url, width=100)
                        st.write(book_name)
        else:
            st.write("No recommendations found.")
    else:
        st.write("Add books to your wishlist to get recommendations.")

# Home page displaying all books
def home_page():
    st.image('logo.jpeg', width=250)
    books = fetch_books()
    display_books(books)

# Wishlist page displaying books added to wishlist
def wishlist_page():
    st.image('logo.jpeg', width=250)
    st.title("Wishlist")
    view_list('wishlist')

# Cart page displaying books added to cart
def cart_page():
    st.image('logo.jpeg', width=250)
    st.title("Cart")
    view_list('cart')

# Search page to search for books
def search_page():
    st.image('logo.jpeg', width=250)
    st.title("Search Books")
    search_query = st.text_input("Enter search term...")
    if search_query:
        books = fetch_books(search_query)
        display_books(books)
    else:
        st.write("Please enter a search term.")

# Recommended books page
def recommended_page():
    st.image('logo.jpeg', width=250)
    st.title("Recommended Books")
    recommend_books()

# Main app
def main():
    # Sidebar with app name and tagline
    st.sidebar.image('logo.jpeg', width=250)

    # Sidebar navigation
    page = st.sidebar.radio("Navigate", ["Home", "Search", "Wishlist", "Cart", "Recommended"])

    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Search":
        search_page()
    elif page == "Wishlist":
        wishlist_page()
    elif page == "Cart":
        cart_page()
    elif page == "Recommended":
        recommended_page()

if __name__ == "_main_":
    main()