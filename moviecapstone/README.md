# 🎬 Movie Recommendation System

This **Movie Recommendation System** allows users to find similar movies based on either a movie title or a description of the type of movie they are looking for. It utilizes **Sentence Transformers** for embedding generation and **FAISS** for fast similarity search, which enables real-time recommendations.
## Author

- **[Laçin Boz]([https://github.com/your-github-username](https://github.com/lacinboz))** 


## ⭐ Features

✅ **Search by Movie Title** – Enter a movie title, and the system will return the top 5 most similar movies.  
✅ **Describe a Movie** – Describe what kind of movie you are looking for, and the system will suggest the best matching movies.  
✅ **Fast Similarity Search** – Uses **FAISS** for high-speed search through movie embeddings.  
✅ **Interactive Web Interface** – Built with **Streamlit**, providing an easy-to-use UI for movie discovery.  

---

## 🚀 How to Run

### **1. Clone the Repository**
   - First, clone the GitHub repository and navigate to the project folder:
     ```bash
     git clone https://github.com/ahmedavid/sdr_capstone.git
     cd sdr_capstone/moviecapstone
     ```

### **2. Install Dependencies**
   - Install the required libraries:
     ```bash
     pip install faiss-cpu pandas numpy sentence-transformers streamlit
     ```

### **3. Generate the FAISS Index (If Not Already Done)**
   - If the **FAISS index** has not been created yet, run the **index_generator.py** script:
     ```bash
     python index_generator.py
     ```
   - This step computes the embeddings and saves them in a **FAISS index** for faster similarity searches.

### **4. Start the Streamlit App**
   - Run the Streamlit app by executing the following command:
     ```bash
     streamlit run app.py
     ```
   - This will open the application in your default web browser, allowing you to search for movies.

---

## 🛠️ What You Need

📌 **Libraries Required:**
   - **faiss-cpu** – For fast similarity search
   - **pandas** – For data manipulation
   - **numpy** – For numerical operations
   - **sentence-transformers** – For generating movie embeddings
   - **streamlit** – For the interactive web interface

---

## ⚙️ How Project Works 

### **Step 1: Data Preparation**
- The dataset, **`netflix_titles.csv`**, contains movie details such as **title, genres, director, cast, and descriptions**.
- The code generates a **textual representation** for each movie, which combines all relevant information.

### **Step 2: Generating Embeddings**
- **Sentence Transformers** converts movie descriptions into **high-dimensional vectors (embeddings)**.
- Each movie's textual data is mapped into a **numerical representation** that captures the meaning and context of the movie.

### **Step 3: FAISS Indexing**
- FAISS (Facebook AI Similarity Search) is a library optimized for **fast nearest neighbor search**.
- The embeddings are stored in a **vector index**, where each movie is represented as a point in a high-dimensional space.
- When a user searches for a movie or enters a description:
  - The input is **converted into an embedding** using the same Sentence Transformers model.
  - FAISS **compares the new embedding** to the indexed movie embeddings using **cosine similarity** (or Euclidean distance).
  - The system **retrieves the top 5 most similar movies** based on the distance in the vector space.

### **Step 4: Returning Recommendations**
- The system returns **the closest matching movies** based on their embeddings.
- Users can refine their search for better recommendations.

---
![Describe A Mpvie](https://github.com/ahmedavid/sdr_capstone/blob/main/moviecapstone/example1.png)

![Search by Movie](https://github.com/ahmedavid/sdr_capstone/blob/main/moviecapstone/example2.png)






