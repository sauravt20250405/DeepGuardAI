import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash

class DeepGuardDB:
    def __init__(self, host='localhost', user='root', password='Luck@492025', database='DeepGuard_AI'):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor(dictionary=True)

    # ====================== USER MANAGEMENT ======================
    
    def create_user(self, username, password, role='user'):
        """Register a new user with a hashed password."""
        pw_hash = generate_password_hash(password)
        query = "INSERT INTO Users (username, password_hash, role) VALUES (%s, %s, %s)"
        self.cursor.execute(query, (username, pw_hash, role))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_user_by_username(self, username):
        """Fetch a user record by username."""
        query = "SELECT * FROM Users WHERE username = %s"
        self.cursor.execute(query, (username,))
        return self.cursor.fetchone()
    
    def verify_user(self, username, password):
        """Verify credentials. Returns user dict if valid, None otherwise."""
        user = self.get_user_by_username(username)
        if user and check_password_hash(user['password_hash'], password):
            return user
        return None

    def get_all_users(self):
        """Admin: fetch all registered users."""
        query = "SELECT user_id, username, role, created_at FROM Users ORDER BY created_at DESC"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    # ====================== MEDIA LOGGING ======================

    def log_media(self, filename, file_size, user_id=None):
        """Log an uploaded media file, optionally tied to a user."""
        query = "INSERT INTO MediaMetadata (user_id, filename, file_size_mb, upload_date) VALUES (%s, %s, %s, NOW())"
        self.cursor.execute(query, (user_id, filename, file_size))
        self.conn.commit()
        return self.cursor.lastrowid

    def save_analysis(self, media_id, confidence, verdict, model_name):
        query = """INSERT INTO AnalysisResults (media_id, confidence_score, verdict, model_used) 
                   VALUES (%s, %s, %s, %s)"""
        self.cursor.execute(query, (media_id, confidence, verdict, model_name))
        self.conn.commit()
        return self.cursor.lastrowid 

    def log_artifact(self, analysis_id, frame_number, artifact_type):
        query = "INSERT INTO DetectionArtifacts (analysis_id, frame_number, artifact_type) VALUES (%s, %s, %s)"
        self.cursor.execute(query, (analysis_id, frame_number, artifact_type))
        self.conn.commit()

    # ====================== LOG RETRIEVAL ======================

    def get_logs_by_user(self, user_id):
        """Get scan logs only for a specific user."""
        query = """
            SELECT m.media_id, m.filename, m.file_size_mb, m.upload_date,
                   a.confidence_score, a.verdict, a.model_used
            FROM MediaMetadata m
            LEFT JOIN AnalysisResults a ON m.media_id = a.media_id
            WHERE m.user_id = %s
            ORDER BY m.upload_date DESC
        """
        self.cursor.execute(query, (user_id,))
        return self.cursor.fetchall()

    def get_all_logs(self):
        """Admin: Get ALL scan logs from every user."""
        query = """
            SELECT m.media_id, m.filename, m.file_size_mb, m.upload_date,
                   a.confidence_score, a.verdict, a.model_used,
                   u.username
            FROM MediaMetadata m
            LEFT JOIN AnalysisResults a ON m.media_id = a.media_id
            LEFT JOIN Users u ON m.user_id = u.user_id
            ORDER BY m.upload_date DESC
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        if self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            print("MySQL connection is closed")