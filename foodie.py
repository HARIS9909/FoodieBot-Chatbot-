# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import ast

# # Load the dataset
# df = pd.read_csv("C:/Users/Dell/My Chatbot/recipes.csv")  # Replace with your actual dataset path

# # Check for missing values in key columns
# df = df.fillna('')

# # Ensure that 'RecipeIngredientParts' and 'RecipeIngredientQuantities' are strings
# df['RecipeIngredientParts'] = df['RecipeIngredientParts'].astype(str)
# df['RecipeIngredientQuantities'] = df['RecipeIngredientQuantities'].astype(str)

# # Combine ingredients and quantities into a single string
# df['ingredients_str'] = df['RecipeIngredientParts'] + " " + df['RecipeIngredientQuantities']

# # Convert the ingredients columns from string 'list' format to actual lists
# def convert_to_list(value):
#     try:
#         # Remove the 'c(' and ')' parts, then split by commas
#         value = value.replace('c(', '').replace(')', '')
#         # Convert to list
#         return [item.strip().strip('"') for item in value.split(',')]
#     except Exception as e:
#         return []

# df['RecipeIngredientParts'] = df['RecipeIngredientParts'].apply(convert_to_list)
# df['RecipeIngredientQuantities'] = df['RecipeIngredientQuantities'].apply(convert_to_list)

# # Function to get recipe recommendations based on user input
# def find_recipes(user_input):
#     # Check if the input seems like a dish name (multiple words, no comma)
#     if len(user_input.split()) > 1 and "," not in user_input:
#         # Treat it as a dish name and search in the 'Name' column
#         matched_recipes = df[df['Name'].str.contains(user_input, case=False, na=False)]
#     else:
#         # Treat it as ingredients, using TF-IDF
#         vectorizer = TfidfVectorizer(stop_words='english')
#         tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])
        
#         # Transform the user input (ingredients) into the same vector space
#         user_tfidf = vectorizer.transform([user_input])
        
#         # Calculate cosine similarities between user input and recipe ingredients
#         cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        
#         # Get the top 5 most similar recipes
#         top_indices = cosine_sim.argsort()[-5:][::-1]
#         matched_recipes = df.iloc[top_indices]
    
#     # Display the top matching recipes
#     recommended_recipes = []
#     for _, recipe in matched_recipes.iterrows():
#         # Combine the ingredients quantities and parts into a full list
#         ingredients = list(zip(recipe['RecipeIngredientQuantities'], recipe['RecipeIngredientParts']))
#         formatted_ingredients = ', '.join([f"{qty} {part}" for qty, part in ingredients])
        
#         # Handle missing or empty description
#         description = recipe['Description'] if recipe['Description'] != 'nan' else 'No description available'
        
#         # Get the instructions (we use RecipeInstructions column)
#         instructions = recipe['RecipeInstructions'] if recipe['RecipeInstructions'] != 'nan' else 'No instructions available'
        
#         recommended_recipes.append({
#             "name": recipe['Name'],
#             "cook_time": recipe['CookTime'],
#             "prep_time": recipe['PrepTime'],
#             "total_time": recipe['TotalTime'],
#             "description": description,
#             "ingredients": formatted_ingredients,
#             "instructions": instructions
#         })
#     return recommended_recipes

# # Function to interact with the user and get recipe suggestions
# def foodie_chatbot():
#     print("🍎 Welcome to FoodieBot! I'm here to help you find recipes.")
#     print("You can either tell me the ingredients you have or the dish name you want to make.")
    
#     # Repeated interaction loop
#     while True:
#         # Get user input
#         user_input = input("\nWhat ingredients or dish name do you have? (comma-separated ingredients or dish name): ").lower()
        
#         # Get recipe recommendations based on user input
#         print(f"\n🔍 Searching for recipes for: {user_input}...")
#         results = find_recipes(user_input)
        
#         if results:
#             print(f"\n🎉 I found {len(results)} matching recipes!\n")
#             for idx, recipe in enumerate(results):
#                 print(f"🍳 Recipe {idx+1}: {recipe['name']}")
#                 print(f"   ⏱️ Cook time: {recipe['cook_time']} | Prep time: {recipe['prep_time']} | Total time: {recipe['total_time']}")
#                 print(f"   📊 Description: {recipe['description']}")
#                 print(f"   ✅ Ingredients: {recipe['ingredients']}")
#                 print(f"   📝 Instructions: {recipe['instructions']}")
#                 print("-" * 40)  # Separator for readability
            
#             # Ask if the user wants to refine the search or continue
#             next_action = input("\nDo you want to refine your search (e.g., by dish type) or search again? (y/n): ").lower()
#             if next_action != 'y':
#                 print("👋 Goodbye!")
#                 break
#         else:
#             print("Sorry, no matching recipes found. Try again with different ingredients or dish names.")

# # Start the chatbot
# if __name__ == "__main__":
#     foodie_chatbot()


# import json
# import random
# import re
# import streamlit as st
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # ------------------------------
# # PAGE CONFIGURATION
# # ------------------------------
# st.set_page_config(
#     page_title="FoodieBot 🍽️",
#     page_icon="🍽️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ------------------------------
# # CUSTOM CSS FOR MODERN STYLING
# # ------------------------------
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: 700;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         color: #666;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .chat-container {
#         background: #f8f9fa;
#         border-radius: 15px;
#         padding: 20px;
#         margin: 10px 0;
#         border: 1px solid #e9ecef;
#     }
#     .user-message {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 12px 18px;
#         border-radius: 18px 18px 5px 18px;
#         margin: 5px 0;
#         max-width: 80%;
#         margin-left: auto;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     }
#     .bot-message {
#         background: white;
#         color: #333;
#         padding: 12px 18px;
#         border-radius: 18px 18px 18px 5px;
#         margin: 5px 0;
#         max-width: 80%;
#         border: 1px solid #e9ecef;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.05);
#     }
#     .recipe-section {
#         background: white;
#         padding: 15px;
#         border-radius: 10px;
#         margin: 10px 0;
#         border-left: 4px solid #667eea;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#     }
#     .ingredient-list {
#         background: #f8f9fa;
#         padding: 15px;
#         border-radius: 8px;
#         margin: 10px 0;
#     }
#     .instruction-list {
#         background: #fff9f2;
#         padding: 15px;
#         border-radius: 8px;
#         margin: 10px 0;
#     }
#     .stButton button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 25px;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
#     .stButton button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
#     }
#     .quick-action-btn {
#         background: white !important;
#         color: #667eea !important;
#         border: 2px solid #667eea !important;
#         margin: 5px;
#     }
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
#     }
# </style>
# """, unsafe_allow_html=True)

# # ------------------------------
# # INITIALIZE SESSION STATE
# # ------------------------------
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'last_recipe' not in st.session_state:
#     st.session_state.last_recipe = None
# if 'initialized' not in st.session_state:
#     st.session_state.initialized = False

# # ------------------------------
# # MODEL FUNCTIONS (Same as your original code)
# # ------------------------------
# def clean_sentence(sentence):
#     return re.findall(r'\b\w+\b', sentence.lower())

# def load_model():
#     # Load intents
#     with open("intents.json", encoding="utf-8") as file:
#         data = json.load(file)

#     training_sentences = []
#     training_labels = []
#     responses = {}

#     for intent in data["intents"]:
#         for example in intent["examples"]:
#             training_sentences.append(example)
#             training_labels.append(intent["intent"])
#         responses[intent["intent"]] = intent["responses"]

#     # Vectorizer
#     vectorizer = CountVectorizer(tokenizer=clean_sentence)
#     X_train = vectorizer.fit_transform(training_sentences)

#     # Model
#     model = MultinomialNB()
#     model.fit(X_train, training_labels)
    
#     return model, vectorizer, responses

# # Load model components
# model, vectorizer, responses = load_model()

# # Dish detection
# dishes = [
#     "biryani", "nihari", "haleem",
#     "karahi", "gajar ka halwa", "halwa", "chicken karahi","pulao"
# ]

# def detect_dish(text):
#     text = text.lower()
#     for dish in dishes:
#         if dish in text:
#             key = dish.replace(" ", "_")
#             return f"recipe_{key}"
#     return None

# def predict_intent(text):
#     # Dish detection first
#     dish_intent = detect_dish(text)
#     if dish_intent:
#         return dish_intent
    
#     # ML prediction
#     X_input = vectorizer.transform([text])
#     probs = model.predict_proba(X_input)[0]
#     max_prob = max(probs)
#     label = model.predict(X_input)[0]
    
#     # LOW CONFIDENCE → unknown
#     if max_prob < 0.45:
#         return "unknown"
    
#     return label

# def get_response(intent, mode="full"):
#     # Handling Simple Intents: Greeting, Help, Thanks, etc.
#     if intent in ["greet", "thanks", "help", "about", "goodbye"]:
#         return random.choice(responses[intent])["instructions"]

#     # Unknown
#     if intent == "unknown":
#         return "I'm still learning. Try asking for a recipe like 'How to make Biryani?'"

#     # Recipe
#     if intent.startswith("recipe_"):
#         response = random.choice(responses[intent])

#         ingredients = response["ingredients"]
#         instructions = response["instructions"]
        
#         # Only ingredients
#         if mode == "ingredients":
#             return f"📝 **Ingredients:**\n{ingredients}"

#         # Only instructions
#         elif mode == "steps":
#             return f"👨‍🍳 **Steps:**\n{instructions}"

#         # Full recipe
#         else:
#             return f"📝 **Ingredients:**\n{ingredients}\n\n👨‍🍳 **Instructions:**\n{instructions}"

#     # Fallback
#     return "Sorry, something went wrong."

# # ------------------------------
# # SIDEBAR
# # ------------------------------
# with st.sidebar:
#     st.markdown("## 🍽️ FoodieBot")
#     st.markdown("---")
#     st.markdown("### Quick Actions")
    
#     # Quick recipe buttons
#     st.markdown("**Popular Recipes:**")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("🍛 Biryani", use_container_width=True, key="biryani"):
#             st.session_state.chat_history.append(("user", "How to make biryani?"))
#             intent = predict_intent("How to make biryani?")
#             response = get_response(intent)
#             st.session_state.chat_history.append(("bot", response))
#             st.session_state.last_recipe = intent
#             st.rerun()
        
#         if st.button("🍗 Karahi", use_container_width=True, key="karahi"):
#             st.session_state.chat_history.append(("user", "Chicken karahi recipe"))
#             intent = predict_intent("Chicken karahi recipe")
#             response = get_response(intent)
#             st.session_state.chat_history.append(("bot", response))
#             st.session_state.last_recipe = intent
#             st.rerun()
    
#     with col2:
#         if st.button("🍲 Nihari", use_container_width=True, key="nihari"):
#             st.session_state.chat_history.append(("user", "Tell me about nihari"))
#             intent = predict_intent("Tell me about nihari")
#             response = get_response(intent)
#             st.session_state.chat_history.append(("bot", response))
#             st.session_state.last_recipe = intent
#             st.rerun()
        
#         if st.button("🍮 Halwa", use_container_width=True, key="halwa"):
#             st.session_state.chat_history.append(("user", "Gajar ka halwa recipe"))
#             intent = predict_intent("Gajar ka halwa recipe")
#             response = get_response(intent)
#             st.session_state.chat_history.append(("bot", response))
#             st.session_state.last_recipe = intent
#             st.rerun()
    
#     st.markdown("---")
#     st.markdown("### Chat Controls")
    
#     if st.button("🔄 Clear Chat", use_container_width=True):
#         st.session_state.chat_history = []
#         st.session_state.last_recipe = None
#         st.rerun()
    
#     if st.button("ℹ️ Help", use_container_width=True):
#         st.session_state.chat_history.append(("user", "help"))
#         intent = predict_intent("help")
#         response = get_response(intent)
#         st.session_state.chat_history.append(("bot", response))
#         st.rerun()
    
#     st.markdown("---")
#     st.markdown("### About")
#     st.markdown("""
#     FoodieBot helps you discover and cook delicious recipes! 
    
#     **Features:**
#     - 🍽️ Recipe discovery
#     - 📝 Ingredients list
#     - 👨‍🍳 Cooking instructions
#     - 💬 Interactive chat
    
#     Ask me about any recipe!
#     """)

# # ------------------------------
# # MAIN INTERFACE
# # ------------------------------
# st.markdown('<div class="main-header">FoodieBot 🍽️</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-header">Your AI Chef Assistant - Discover and Cook Amazing Recipes!</div>', unsafe_allow_html=True)

# # Chat container
# st.markdown("### 💬 Chat with FoodieBot")

# # Display chat history
# chat_container = st.container()
# with chat_container:
#     for sender, message in st.session_state.chat_history:
#         if sender == "user":
#             st.markdown(f'<div class="user-message">👤 You: {message}</div>', unsafe_allow_html=True)
#         else:
#             # Format recipe responses
#             if "**Ingredients:**" in message and "**Instructions:**" in message:
#                 parts = message.split("**Instructions:**")
#                 ingredients_part = parts[0].replace("**Ingredients:**", "").strip()
#                 instructions_part = parts[1].strip()
                
#                 st.markdown('<div class="bot-message">', unsafe_allow_html=True)
#                 st.markdown("**🤖 FoodieBot:**")
                
#                 # Ingredients section
#                 st.markdown("**📝 Ingredients:**")
#                 st.markdown(f'<div class="ingredient-list">{ingredients_part}</div>', unsafe_allow_html=True)
                
#                 # Instructions section
#                 st.markdown("**👨‍🍳 Instructions:**")
#                 st.markdown(f'<div class="instruction-list">{instructions_part}</div>', unsafe_allow_html=True)
                
#                 st.markdown('</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="bot-message">🤖 FoodieBot: {message}</div>', unsafe_allow_html=True)

# # Quick action buttons for last recipe
# # if st.session_state.last_recipe:
# #     st.markdown("### 🔄 Quick Actions")
# #     col1, col2, col3 = st.columns(3)
    
# #     with col1:
# #         if st.button("📋 Show Ingredients Only", use_container_width=True):
# #             response = get_response(st.session_state.last_recipe, mode="ingredients")
# #             st.session_state.chat_history.append(("user", "Show ingredients"))
# #             st.session_state.chat_history.append(("bot", response))
# #             st.rerun()
    
# #     with col2:
# #         if st.button("👨‍🍳 Show Steps Only", use_container_width=True):
# #             response = get_response(st.session_state.last_recipe, mode="steps")
# #             st.session_state.chat_history.append(("user", "Show steps"))
# #             st.session_state.chat_history.append(("bot", response))
# #             st.rerun()
    
# #     with col3:
# #         if st.button("🍽️ Show Full Recipe", use_container_width=True):
# #             response = get_response(st.session_state.last_recipe)
# #             st.session_state.chat_history.append(("user", "Show full recipe"))
# #             st.session_state.chat_history.append(("bot", response))
# #             st.rerun()

# # Chat input
# st.markdown("---")
# user_input = st.chat_input("Ask me about recipes... (e.g., 'How to make biryani?')")

# if user_input:
#     # Add user message to chat history
#     st.session_state.chat_history.append(("user", user_input))
    
#     # Get bot response
#     if any(x in user_input for x in ["also", "more", "again"]) and st.session_state.last_recipe:
#         intent = st.session_state.last_recipe
#     else:
#         intent = predict_intent(user_input)
#         if intent.startswith("recipe_"):
#             st.session_state.last_recipe = intent
    
#     # Determine mode
#     if "ingredient" in user_input:
#         response = get_response(intent, mode="ingredients")
#     elif "step" in user_input or "instruction" in user_input:
#         response = get_response(intent, mode="steps")
#     else:
#         response = get_response(intent)
    
#     # Add bot response to chat history
#     st.session_state.chat_history.append(("bot", response))
#     st.rerun()

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: #666;'>"
#     "Built with ❤️ using Streamlit | FoodieBot v1.0"
#     "</div>",
#     unsafe_allow_html=True
# )



# import json
# import random
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # ------------------------------
# # CLEANING FUNCTION
# # ------------------------------
# def clean_sentence(sentence):
#     return re.findall(r'\b\w+\b', sentence.lower())

# # ------------------------------
# # LOAD INTENTS
# # ------------------------------
# with open("intents.json", encoding="utf-8") as file:
#     data = json.load(file)

# training_sentences = []
# training_labels = []
# responses = {}

# for intent in data["intents"]:
#     for example in intent["examples"]:
#         training_sentences.append(example)
#         training_labels.append(intent["intent"])
#     responses[intent["intent"]] = intent["responses"]

# # ------------------------------
# # VECTORIZER
# # ------------------------------
# vectorizer = CountVectorizer(tokenizer=clean_sentence)
# X_train = vectorizer.fit_transform(training_sentences)

# # ------------------------------
# # MODEL TRAINING
# # ------------------------------
# model = MultinomialNB()
# model.fit(X_train, training_labels)

# # ------------------------------
# # DISH EXTRACTION (NLP-Like)
# # ------------------------------
# dishes = [
#     "biryani", "nihari", "haleem",
#     "karahi", "gajar ka halwa", "halwa", "chicken karahi"
# ]

# def detect_dish(text):
#     text = text.lower()
#     for dish in dishes:
#         if dish in text:
#             key = dish.replace(" ", "_")
#             return f"recipe_{key}"
#     return None

# # ------------------------------
# # PREDICT INTENT WITH CONFIDENCE
# # ------------------------------
# def predict_intent(text):
#     # Dish detection first
#     dish_intent = detect_dish(text)
#     if dish_intent:
#         return dish_intent
    
#     # ML prediction
#     X_input = vectorizer.transform([text])
#     probs = model.predict_proba(X_input)[0]
#     max_prob = max(probs)
#     label = model.predict(X_input)[0]
    
#     # LOW CONFIDENCE → unknown
#     if max_prob < 0.45:
#         return "unknown"
    
#     return label

# # ------------------------------
# # GENERATE RESPONSE
# # ------------------------------
# def get_response(intent, last_recipe=None, mode="full"):
#     # Handling Simple Intents: Greeting, Help, Thanks, etc.
#     if intent in ["greet", "thanks", "help", "about", "goodbye"]:
#         return random.choice(responses[intent])["instructions"]

#     # Unknown
#     if intent == "unknown":
#         return "I'm still learning. Try asking for a recipe like 'How to make Biryani?'"

#     # Recipe
#     if intent.startswith("recipe_"):
#         response = random.choice(responses[intent])

#         ingredients = response["ingredients"]
#         instructions = response["instructions"]
        
#         # Only ingredients
#         if mode == "ingredients":
#             return f"📝 **Ingredients:**\n{ingredients}"

#         # Only instructions
#         elif mode == "steps":
#             return f"👨‍🍳 **Steps:**\n{instructions}"

#         # Full recipe
#         else:
#             return f"📝 **Ingredients:**\n{ingredients}\n\n👨‍🍳 **Instructions:**\n{instructions}"

#     # Fallback
#     return "Sorry, something went wrong."

# # ------------------------------
# # CHATBOT
# # ------------------------------
# def chatbot():
#     print("Hello! I'm your FoodieBot. Ask me any recipe 🍽️")

#     last_intent = None  # memory
#     last_recipe = None  # to store the last recipe

#     while True:
#         user_input = input("You: ").lower()

#         # Exit command
#         if user_input in ["bye", "quit", "exit"]:
#             print("Bot: Goodbye! Have a great day! 👋")
#             break

#         # CONTEXT HANDLING
#         if any(x in user_input for x in ["also", "more", "again"]) and last_recipe:
#             # If user asks for more, repeat the last recipe
#             intent = last_recipe
#         else:
#             intent = predict_intent(user_input)
#             if intent.startswith("recipe_"):
#                 last_recipe = intent  # Store the last recipe

#         # INGREDIENTS ONLY
#         if "ingredient" in user_input:
#             response = get_response(last_recipe, mode="ingredients")
#         # STEPS ONLY
#         elif "step" in user_input or "instruction" in user_input:
#             response = get_response(last_recipe, mode="steps")
#         else:
#             response = get_response(intent)

#         print(f"Bot: {response}")

# # ------------------------------
# # RUN CHATBOT
# # ------------------------------
# chatbot()

# import json
# import random
# import re
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import LabelEncoder

# # Download stopwords if not already downloaded
# nltk.download('stopwords')

# # ------------------------------
# # CLEANING FUNCTION
# # ------------------------------
# def clean_sentence(sentence):
#     stop_words = set(stopwords.words('english'))
#     words = re.findall(r'\b\w+\b', sentence.lower())
#     return [word for word in words if word not in stopwords.words('english')]

# # ------------------------------
# # LOAD INTENTS
# # ------------------------------
# def load_intents():
#     with open("intents.json", encoding="utf-8") as file:
#         data = json.load(file)
#     return data

# # ------------------------------
# # TRAIN THE MODEL
# # ------------------------------
# def train_model(data):
#     training_sentences = []
#     training_labels = []
#     responses = {}

#     # Extract recipes dynamically from the JSON data
#     def get_recipes_from_intents(data):
#         recipes = []
#         for intent in data["intents"]:
#             if intent["intent"].startswith("recipe_"):
#                 recipe_name = intent["intent"].replace("recipe_", "").replace("_", " ")
#                 recipes.append(recipe_name.lower())  # Ensure it's in lowercase for matching
#         return recipes

#     # Get all recipes from the intents
#     recipes = get_recipes_from_intents(data)

#     # Prepare training data
#     for intent in data["intents"]:
#         for example in intent["examples"]:
#             training_sentences.append(example)
#             training_labels.append(intent["intent"])  # Use the actual intent name as the label
#         responses[intent["intent"]] = intent["responses"]

#     # ------------------------------
#     # VECTORIZER AND ENCODER
#     # ------------------------------
#     vectorizer = TfidfVectorizer(tokenizer=clean_sentence, ngram_range=(1, 2))  # Using TF-IDF with bigrams
#     label_encoder = LabelEncoder()
#     training_labels_encoded = label_encoder.fit_transform(training_labels)

#     # ------------------------------
#     # MODEL TRAINING WITH RANDOM FOREST
#     # ------------------------------
#     model = make_pipeline(vectorizer, RandomForestClassifier(n_estimators=100))  # Using RandomForestClassifier
#     model.fit(training_sentences, training_labels_encoded)
    
#     return model, label_encoder, responses

# # ------------------------------
# # PREDICT INTENT WITH CONFIDENCE
# # ------------------------------
# def predict_intent(text, model, label_encoder, threshold=0.60):
#     # ML prediction
#     probs = model.predict_proba([text])[0]
#     max_prob = max(probs)
#     label_index = probs.argmax()
#     label = label_encoder.inverse_transform([label_index])[0]
    
#     # If confidence is below threshold, return "unknown"
#     if max_prob < threshold:
#         return "unknown", max_prob
    
#     return label, max_prob

# # ------------------------------
# # GENERATE RESPONSE
# # ------------------------------
# def get_response(intent, confidence, responses, last_recipe=None, mode="full"):
#     # Handling Simple Intents: Greeting, Help, Thanks, etc.
#     if intent in ["greet", "thanks", "help", "about", "goodbye"]:
#         return f"Confidence: {confidence:.2f} - {random.choice(responses[intent])['instructions']}"

#     # Unknown (fallback if confidence is below threshold)
#     if intent == "unknown":
#         return f"Confidence: {confidence:.2f} - I'm still learning. Try asking for a recipe like 'How to make Biryani?'"

#     # Recipe
#     if intent.startswith("recipe_"):
#         response = random.choice(responses[intent])

#         ingredients = response["ingredients"]
#         instructions = response["instructions"]
        
#         # Only ingredients
#         if mode == "ingredients":
#             return f"Confidence: {confidence:.2f} - 📝 **Ingredients:**\n{ingredients}"

#         # Only instructions
#         elif mode == "steps":
#             return f"Confidence: {confidence:.2f} - 👨‍🍳 **Steps:**\n{instructions}"

#         # Full recipe
#         else:
#             return f"Confidence: {confidence:.2f} - 📝 **Ingredients:**\n{ingredients}\n\n👨‍🍳 **Instructions:**\n{instructions}"

#     # Fallback
#     return f"Confidence: {confidence:.2f} - Sorry, something went wrong."

# # ------------------------------
# # CHATBOT
# # ------------------------------
# def chatbot():
#     print("Hello! I'm your FoodieBot. Ask me any recipe 🍽️")

#     # Load intents and train the model on every server start
#     data = load_intents()
#     model, label_encoder, responses = train_model(data)

#     last_intent = None  # memory
#     last_recipe = None  # to store the last recipe

#     while True:
#         user_input = input("You: ").lower()

#         # Exit command
#         if user_input in ["bye", "quit", "exit"]:
#             print("Bot: Goodbye! Have a great day! 👋")
#             break

#         # CONTEXT HANDLING
#         if any(x in user_input for x in ["also", "more", "again"]) and last_recipe:
#             # If user asks for more, repeat the last recipe
#             intent = last_recipe
#             confidence = 1.0  # Full confidence since we are repeating the last recipe
#         else:
#             intent, confidence = predict_intent(user_input, model, label_encoder, threshold=0.60)  # Adjust threshold here
#             if intent.startswith("recipe_"):
#                 last_recipe = intent  # Store the last recipe

#         # INGREDIENTS ONLY
#         if "ingredient" in user_input:
#             response = get_response(last_recipe, confidence, responses, mode="ingredients")
#         # STEPS ONLY
#         elif "step" in user_input or "instruction" in user_input:
#             response = get_response(last_recipe, confidence, responses, mode="steps")
#         else:
#             response = get_response(intent, confidence, responses)

#         print(f"Bot: {response}")

# # ------------------------------
# # RUN CHATBOT
# # ------------------------------
# chatbot()


import json
import random
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import joblib
import os
import time

# --- NLTK DOWNLOAD ---
# Ensure stopwords are downloaded for the cleaning function
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ------------------------------
# 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# ------------------------------
st.set_page_config(
    page_title="FoodieBot 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# 2. FILE CONSTANTS & GLOBALS
# ------------------------------
MODEL_FILENAME = "chatbot_model.pkl"
INTENTS_FILENAME = "intents.json"
DEFAULT_THRESHOLD = 0.50

# ------------------------------
# 3. CUSTOM CSS FOR MODERN STYLING
# ------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 5px 18px;
        margin: 5px 0;
        max-width: 60%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .bot-message {
        background: white;
        color: #333;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 5px;
        margin: 5px 0;
        max-width: 60%;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .recipe-section {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .ingredient-list {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .instruction-list {
        background: #fff9f2;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .quick-action-btn {
        background: white !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
        margin: 5px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 4. MODEL FUNCTIONS
# ------------------------------

def clean_sentence(sentence):
    """Tokenizes and removes stop words."""
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', sentence.lower())
    return [word for word in words if word not in stop_words]

def load_intents():
    """Load intents from JSON file."""
    with open(INTENTS_FILENAME, encoding="utf-8") as file:
        return json.load(file)

def is_intents_updated():
    """Check if intents.json is newer than the model."""
    if not os.path.exists(MODEL_FILENAME):
        return True
    try:
        intents_mtime = os.path.getmtime(INTENTS_FILENAME)
        model_mtime = os.path.getmtime(MODEL_FILENAME)
        return intents_mtime > model_mtime
    except FileNotFoundError:
        return True

def get_recipes_from_intents(data):
    """Extract recipe names from intents."""
    recipes = []
    for intent in data["intents"]:
        if intent["intent"].startswith("recipe_"):
            recipe_name = intent["intent"].replace("recipe_", "").replace("_", " ")
            recipes.append(recipe_name.lower())
    return recipes

def train_model(data):
    """Train model using Random Forest and TF-IDF."""
    st.info("Training model...")
    training_sentences = []
    training_labels = []
    responses = {}

    for intent in data["intents"]:
        for example in intent["examples"]:
            training_sentences.append(example)
            training_labels.append(intent["intent"])
        responses[intent["intent"]] = intent["responses"]

    # Shuffle data
    training_sentences, training_labels = shuffle(training_sentences, training_labels, random_state=42)

    # Vectorizer and Encoder
    vectorizer = TfidfVectorizer(tokenizer=clean_sentence, ngram_range=(1,2))
    label_encoder = LabelEncoder()
    training_labels_encoded = label_encoder.fit_transform(training_labels)

    # Train model
    model = make_pipeline(vectorizer, RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(training_sentences, training_labels_encoded)

    # Save
    joblib.dump((model, label_encoder, responses), MODEL_FILENAME)
    st.success("Model trained and saved successfully.")
    return model, label_encoder, responses

@st.cache_resource(show_spinner="Loading/Training Model...")
def load_and_train_model():
    """Load or train the model depending on file modification times."""
    if is_intents_updated():
        data = load_intents()
        return train_model(data)
    else:
        st.success("Loaded pre-trained model.")
        return joblib.load(MODEL_FILENAME)

# Load model
try:
    model, label_encoder, responses = load_and_train_model()
except Exception as e:
    st.error(f"Error loading or training model: {e}")
    st.stop()

# ------------------------------
# 5. PREDICT & RESPONSE
# ------------------------------

def predict_intent(text, model, label_encoder, threshold=DEFAULT_THRESHOLD):
    probs = model.predict_proba([text])[0]
    max_prob = max(probs)
    label_index = probs.argmax()
    label = label_encoder.inverse_transform([label_index])[0]

    if max_prob < threshold:
        return "unknown", max_prob
    return label, max_prob

def get_response(intent, confidence, responses, mode="full"):
    """Generate response based on intent and mode."""
    # Simple intents
    if intent in ["greet", "thanks", "help", "about", "goodbye"]:
        message = random.choice(responses[intent])['instructions']
        return f"Confidence: {confidence:.2f} - {message}"

    # Unknown
    if intent == "unknown":
        return f"Confidence: {confidence:.2f} - I'm still learning. Try asking for a recipe like 'How to make Biryani?'"

    # Recipe
    if intent.startswith("recipe_"):
        response = random.choice(responses[intent])
        ingredients = response["ingredients"]
        instructions = response["instructions"]
        header = f"Confidence: {confidence:.2f} - "

        if mode == "ingredients":
            return f"{header}📝 **Ingredients:**\n{ingredients}"
        elif mode == "steps":
            return f"{header}👨‍🍳 **Steps:**\n{instructions}"
        else:
            return f"{header}📝 **Ingredients:**\n{ingredients}\n\n👨‍🍳 **Instructions:**\n{instructions}"

    return f"Confidence: {confidence:.2f} - Sorry, something went wrong."

# ------------------------------
# 6. SESSION STATE
# ------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_recipe' not in st.session_state:
    st.session_state.last_recipe = None

# ------------------------------
# 7. PROCESS USER INPUT
# ------------------------------
def process_user_input(user_input):
    """Process the user's input and update chat history."""
    user_input_lower = user_input.lower()

    # Context: repeat last recipe
    if any(x in user_input_lower for x in ["also", "more", "again"]) and st.session_state.last_recipe:
        intent = st.session_state.last_recipe
        confidence = 1.0
    else:
        intent, confidence = predict_intent(user_input_lower, model, label_encoder)
        if intent.startswith("recipe_"):
            st.session_state.last_recipe = intent

    # Determine mode
    mode = "full"
    if "ingredient" in user_input_lower:
        mode = "ingredients"
    elif "step" in user_input_lower or "instruction" in user_input_lower:
        mode = "steps"

    # Ensure ingredients/steps fallback to last recipe
    if mode in ["ingredients", "steps"]:
        if intent.startswith("recipe_"):
            intent_for_recipe = intent
        else:
            intent_for_recipe = st.session_state.last_recipe
    else:
        intent_for_recipe = intent

    response = get_response(intent_for_recipe, confidence, responses, mode=mode)
    st.session_state.chat_history.append(("bot", response))

# ------------------------------
# 8. SIDEBAR
# ------------------------------
with st.sidebar:
    st.markdown("## 🍽️ FoodieBot")
    st.markdown("---")
    st.markdown("### Quick Actions")
    st.markdown("**Popular Recipes:**")
    col1, col2 = st.columns(2)

    def sidebar_recipe_button(label, user_text, intent_key, col):
        with col:
            if st.button(label, use_container_width=True, key=intent_key):
                st.session_state.chat_history.append(("user", user_text))
                process_user_input(user_text)
                st.rerun()

    sidebar_recipe_button("🍛 Biryani", "How to make biryani?", "biryani", col1)
    sidebar_recipe_button("🍗 Karahi", "How to make Chicken Karahi?", "karahi", col1)
    sidebar_recipe_button("🍲 Nihari", "How to make Nihari?", "nihari", col2)
    sidebar_recipe_button("🍮 Halwa", "How to make Gajar ka Halwa?", "halwa", col2)

    st.markdown("---")
    st.markdown("### Shakes")
    st.markdown("**Popular Shakes:**")

    col1, col2 = st.columns(2)
    sidebar_recipe_button("🥭 Mango", "How to make Mango Milkshake?", "recipe_mango_lassi", col1)
    sidebar_recipe_button("🍓Strawberry", "How to make Strawberry Milkshake?", "recipe_strawberry_milkshake", col1)
    sidebar_recipe_button("🍫 Chocolate", "How to make Chocolate Shake?", "recipe_chocolate_milkshake", col2)
    sidebar_recipe_button("🥥 Oreo ", "Oreo shake kaise banate hain", "recipe_oreo_shake", col2)

    st.markdown("---")
    st.markdown("### Chat Controls")
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_recipe = None
        st.rerun()
    if st.button("ℹ️ Help", use_container_width=True):
        st.session_state.chat_history.append(("user", "help"))
        process_user_input("help")
        st.rerun()
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    FoodieBot helps you discover and cook delicious recipes! 🍽️
    **Features:**
    - 🍲 Recipe discovery
    - 📝 Ingredients list
    - 👨‍🍳 Cooking instructions
    - 💬 Interactive chat
    Ask me about any recipe!
    """)

# ------------------------------
# 9. MAIN INTERFACE
# ------------------------------
st.markdown('<div class="main-header">FoodieBot 🍽️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your AI Chef Assistant - Discover and Cook Amazing Recipes!</div>', unsafe_allow_html=True)
st.markdown("### 💬 Chat with FoodieBot")

chat_container = st.container()
with chat_container:
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            col_user, col_empty = st.columns([0.8, 0.2])
            with col_user:
                st.markdown(f'<div class="user-message">👤 You: {message}</div>', unsafe_allow_html=True)
        else:
            col_empty, col_bot = st.columns([0.2, 0.8])
            with col_bot:
                if "**Ingredients:**" in message and "**Instructions:**" in message:
                    confidence_part, content_part = message.split(" - ", 1)
                    parts = content_part.split("👨‍🍳 **Instructions:**")
                    ingredients_part = parts[0].replace("📝 **Ingredients:**\n", "").strip()
                    instructions_part = parts[1].strip()
                    st.markdown('<div class="bot-message recipe-section">', unsafe_allow_html=True)
                    st.markdown(f"**🤖 FoodieBot** ({confidence_part})")
                    st.markdown("---")
                    st.markdown("**📝 Ingredients:**")
                    st.markdown(f'<div class="ingredient-list">{ingredients_part}</div>', unsafe_allow_html=True)
                    st.markdown("**👨‍🍳 Instructions:**")
                    st.markdown(f'<div class="instruction-list">{instructions_part}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 FoodieBot: {message}</div>', unsafe_allow_html=True)

# Chat input
st.markdown("---")
user_input = st.chat_input("Ask me about recipes... (e.g., 'How to make biryani?')")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    process_user_input(user_input)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using Streamlit | FoodieBot v1.0"
    "</div>",
    unsafe_allow_html=True
)
