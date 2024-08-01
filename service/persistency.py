from sqlite3 import connect, Connection, Cursor
from typing import List
from dataclasses import dataclass, asdict

@dataclass
class MessageHistory:
    thread_id: str
    role: str
    content: str

    def __init__(self, thread_id: str, role: str, content: str):
        self.thread_id = thread_id
        self.role = role
        self.content = content

    def to_dict(self):
        return asdict(self)
class Persistency():

    _conn: Connection
    _curr: Cursor
    def __init__(self):
        self._connect()
        self._curr.execute("""CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id VARCHAR(250) NOT NULL,
            role VARCHAR(250) NOT NULL,
            content TEXT NOT NULL
        )
        """)

    def _connect(self):
        self._conn = connect("conversation.db")
        self._curr = self._conn.cursor()
    
    def add_message(self, message: MessageHistory):
        self._connect()
        self._curr.execute("""INSERT INTO conversation (thread_id, role, content) VALUES (?, ?, ?)""", (message.thread_id, message.role, message.content))
        self._conn.commit()
        self._conn.close()
    
    # Add multiple messages
    def add_messages(self, messages: List[MessageHistory]):
        self._connect()
        records = [(m.thread_id, m.role, m.content) for m in messages]
        self._curr.executemany("INSERT INTO conversation (thread_id, role, content) VALUES (?,?,?)", records)
        self._conn.commit()
        self._conn.close()

    def get_last_k_messages(self, thread_id, k=3):
        self._connect()
        self._curr.execute("SELECT thread_id, role, content FROM conversation WHERE thread_id = ? ORDER BY id DESC LIMIT ?", (thread_id, k))
        messages = self._curr.fetchall()
        self._conn.commit()
        self._conn.close()

        # parse the list of messages to MessageHistory objects
        return [MessageHistory(thread_id=m[0], role=m[1], content=m[2]) for m in messages]
    
    # get all messages by thread_id
    def get_messages(self, thread_id):
        self._connect()
        self._curr.execute("SELECT thread_id, role, content FROM conversation WHERE thread_id = ?", (thread_id,))
        messages = self._curr.fetchall()
        self._conn.commit()
        self._conn.close()

        # parse the list of messages to MessageHistory objects
        return [MessageHistory(thread_id=m[0], role=m[1], content=m[2]) for m in messages]


    # clear memory for a specific thread
    def clear_memory(self, thread_id):
        self._connect()
        self._curr.execute("DELETE FROM conversation WHERE thread_id = ?", (thread_id,))
        self._conn.commit()
        self._conn.close()