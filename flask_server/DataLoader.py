import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datetime import timedelta
import re
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime


class DataLoader:
    def __init__(self, data_dir="data", qdrant_host="qdrant", qdrant_port=6333):
        self.data_dir = data_dir
        self.file_participants_csv = os.path.join(data_dir, "file_participants.csv")
        self.files_csv = os.path.join(data_dir, "files.csv")
        self.files_df = pd.read_csv(self.files_csv)
        self.users_csv = os.path.join(data_dir, "users.csv")
        self.users_df = pd.read_csv(self.users_csv)
        self.file_participants_df = pd.read_csv(self.file_participants_csv)
        # initialize Qdrant client
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "call_transcripts"
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        # Create collection if not exists
        if self.collection_name not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )

    def _get_pending_files(self, user_id=None):
        if user_id:
            # Get files associated with this user
            file_ids = self.file_participants_df[self.file_participants_df['user_id'] == user_id]['file_id'].unique()
            pending_files = self.files_df[
                (self.files_df['file_id'].isin(file_ids)) & (self.files_df['status'] == "Pending")]
        else:
            pending_files = self.files_df[self.files_df['status'] == "Pending"]
        return pending_files

    def _get_speakers(self, file_id):
        user_ids = self.file_participants_df[self.file_participants_df['file_id'] == file_id]['user_id'].tolist()
        speakers = self.users_df[self.users_df['user_id'].isin(user_ids)]['user_name'].tolist()
        return speakers

    def _parse_timestamp(self, timestamp_str):
        # Parse [mm:ss] format
        match = re.match(r"\[(\d+):(\d+)\]", timestamp_str)
        if match:
            minutes, seconds = int(match.group(1)), int(match.group(2))
            return timedelta(minutes=minutes, seconds=seconds)
        return None

    def _chunk_text(self, text, chunk_minutes=2):
        """
        Splits transcript into chunks based on a time window.
        chunk_minutes: size of each chunk in minutes
        """
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        start_time = None
        window = timedelta(minutes=chunk_minutes)

        for line in lines:
            ts_match = re.match(r"\[(\d+:\d+)\]", line)
            if not ts_match:
                continue
            ts = self._parse_timestamp(ts_match.group(0))
            if start_time is None:
                start_time = ts

            if ts - start_time < window:
                current_chunk.append(line)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [line]
                start_time = ts

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _process_file(self, file_row):
        file_id = file_row['file_id']
        filename = file_row['filename']
        timestamp = file_row['timestamp']
        filepath = os.path.join("./transcript", filename)
        if not os.path.exists(filepath):
            print(f"[Warning] File not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Chunking
        chunks = self._chunk_text(content)

        # Get speakers
        speakers = self._get_speakers(file_id)

        # Create embeddings and push to Qdrant
        vectors = self.model.encode(chunks)

        points = []
        for idx, vec in enumerate(vectors):
            try:
                # Ensure embedding dimension is correct
                if vec is None or len(vec) == 0:
                    print(f"[Warning] Skipping empty vector for chunk {idx}")
                    continue

                metadata = {
                    "call_id": filename,
                    "speakers": speakers,
                    "chunk_index": idx,
                    "timestamp": timestamp,
                    "text_data": chunks[idx]
                }

                # Create Qdrant point
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec.tolist(),
                        payload=metadata
                    )
                )

            except Exception as e:
                # Skip only the problematic chunk
                print(f"[Error] Skipping chunk {idx} due to error: {e}")
                continue

                # Upload only valid points
        if points:
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            print(f"[Info] Uploaded {len(points)} chunks for {filename} to Qdrant")
        else:
            print(f"[Warning] No valid chunks uploaded for {filename}")

        # Update file status
        self.files_df.loc[self.files_df['file_id'] == file_id, 'status'] = "Completed"
        self.files_df.to_csv(self.files_csv, index=False)
        print(f"[Info] File {filename} marked as Completed")


        # for idx, vec in enumerate(vectors):
        #
        #     metadata = {
        #         "call_id": filename,
        #         "speakers": speakers,
        #         "chunk_index": idx,
        #         "timestamp": timestamp,
        #         "text_data": chunks[idx]
        #     }
        #     points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=metadata))
        #
        # self.qdrant.upsert(collection_name=self.collection_name, points=points)
        # print(f"[Info] Uploaded {len(points)} chunks for {filename} to Qdrant")
        #
        # # Update file status
        # self.files_df.loc[self.files_df['file_id'] == file_id, 'status'] = "Completed"
        # self.files_df.to_csv(self.files_csv, index=False)
        # print(f"[Info] File {filename} marked as Completed")

    def process_all_pending(self):
        pending_files = self._get_pending_files()
        print(f"[Info] Processing {len(pending_files)} pending files...")
        for _, row in pending_files.iterrows():
            self._process_file(row)

    def process_user_files(self, user_id):
        pending_files = self._get_pending_files(user_id=int(user_id))
        print(f"[Info] Processing {len(pending_files)} pending files for user_id={user_id}...")
        for _, row in pending_files.iterrows():
            self._process_file(row)

