import os
import boto3
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
from pymongo.server_api import ServerApi
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
import secrets

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# Generate and set your JWT secret key
app.config['JWT_SECRET_KEY'] = secrets.token_hex(32)
jwt = JWTManager(app)

# Initialize AWS Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1')

# Initialize MongoDB client
uri = "mongodb+srv://myappworld1983:883dTAH4kYQqPs0M@petconnect.6tikz8w.mongodb.net/?retryWrites=true&w=majority&appName=petconnect"
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client['petconnecttest']
collection = db['imagevector']

# Function to read image file from S3 and convert to bytes
def load_image_from_s3(s3_path):
    s3 = boto3.client('s3')
    bucket_name = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    response = s3.get_object(Bucket=bucket_name, Key=key)
    return response['Body'].read()

# Function to send image to Rekognition and get labels with confidence
def get_labels_from_image(image_bytes):
    response = rekognition.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=10)
    labels_with_confidence = [{'Name': label['Name'], 'Confidence': label['Confidence']} for label in response['Labels']]
    return labels_with_confidence

# Function to create a dictionary of label confidences
def create_confidence_dict(labels_with_confidence):
    return {label_info['Name']: label_info['Confidence'] for label_info in labels_with_confidence}

# Function to perform clustering
def perform_clustering():
    # Retrieve all documents from MongoDB
    documents = list(collection.find())

    # Create a set of all unique labels
    all_labels = set()
    for document in documents:
        try:
            labels_with_confidence = document['labels']
        except KeyError:
            print(f"Document with missing 'labels' field: {document['_id']}")
            continue
        all_labels.update(label['Name'] for label in labels_with_confidence)

    # Create a matrix of confidences, where each row corresponds to an image and each column to a label
    label_list = list(all_labels)
    confidence_matrix = np.zeros((len(documents), len(label_list)))

    for i, document in enumerate(documents):
        try:
            labels_with_confidence = document['labels']
        except KeyError:
            print(f"Document with missing 'labels' field: {document['_id']}")
            continue
        confidence_dict = create_confidence_dict(labels_with_confidence)
        for label, confidence in confidence_dict.items():
            j = label_list.index(label)
            confidence_matrix[i, j] = confidence

    # Perform K-means clustering on the confidence matrix
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(confidence_matrix)

    # List to store user IDs for each cluster
    cluster_user_ids = [[] for _ in range(3)]

    # Aggregate user IDs for each cluster
    for i, document in enumerate(documents):
        try:
            user_id = document['user_id']
        except KeyError:
            print(f"Document with missing 'user_id' field: {document['_id']}")
            continue
        cluster_label = clusters[i]
        cluster_user_ids[cluster_label].append(user_id)

    # Create a new collection for storing clustered data
    clustered_collection = db['image_clusters']

    # Clear existing data in the collection (optional step)
    clustered_collection.delete_many({})

    # Store clustered data in MongoDB
    for cluster_label, user_ids in enumerate(cluster_user_ids):
        clustered_document = {
            'cluster': cluster_label,
            'user_ids': user_ids
        }
        clustered_collection.insert_one(clustered_document)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    # Replace with actual user verification
    if username != 'test' or password != 'test':
        return jsonify({"msg": "Bad username or password"}), 401
    
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/process-image', methods=['POST'])
@jwt_required()
def process_image():
    data = request.get_json()
    user_id = data['user_id']
    s3_path = data['s3_path']
    
    image_bytes = load_image_from_s3(s3_path)
    labels_with_confidence = get_labels_from_image(image_bytes)
    
    weighted_vector_values = [{'weight': label_info['Confidence'], 'label': label_info['Name']} for label_info in labels_with_confidence]
    
    document = {
        'user_id': user_id,
        'image_path': s3_path,
        'labels': labels_with_confidence,
        'weighted_vector_values': weighted_vector_values
    }.insert_one(document)

    # Perform clustering after processing the image
    perform_clustering()
    
    return jsonify({"message": "Image processed, clustering updated, and data stored successfully."})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
