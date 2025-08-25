import json
import csv


class EventHandler:
    """Handles individual events and maintains CSV data"""

    def __init__(self):
        self.events = []  # store Event dicts
        self.users_set = set()
        self.file_participants = []

    def handle_event(self, event):
        """Process a single event"""
        # Store event
        self.events.append(event)
        # Track users
        for user in event['users']:
            self.users_set.add(user)
        # Track participants
        self.file_participants.append({'filename': event['filename'], 'users': event['users']})

    def write_csvs(self):
        """Generate files.csv, users.csv, file_participants.csv"""
        # Map users to IDs
        user_list = list(self.users_set)
        user_to_id = {user: idx + 1 for idx, user in enumerate(user_list)}

        # files.csv
        with open('data/files.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_id', 'filename', 'status', 'timestamp'])
            for idx, event in enumerate(self.events, 1):
                writer.writerow([idx, event['filename'], 'Pending', event['timestamp']])

        # users.csv
        with open('data/users.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'user_name'])
            for user, uid in user_to_id.items():
                writer.writerow([uid, user])

        # file_participants.csv
        with open('data/file_participants.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_id', 'user_id'])
            for file_idx, fp in enumerate(self.file_participants, 1):
                for user in fp['users']:
                    writer.writerow([file_idx, user_to_id[user]])


class EventSimulator:
    """Simulates reading events from JSON and sending to EventHandler"""

    def __init__(self, json_file, event_handler):
        self.json_file = json_file
        self.event_handler = event_handler

    def run(self):
        """Read events and push to event handler one by one"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        for event in events:
            self.event_handler.handle_event(event)
