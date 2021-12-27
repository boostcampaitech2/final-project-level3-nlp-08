import sqlite3


def create_table():
    conn = sqlite3.connect("db.db")
    c = conn.cursor()

    c.execute(
        """
    CREATE TABLE CLIENT (
        ID varchar(20) PRIMARY KEY
    );
    """
    )

    c.execute(
        """
    CREATE TABLE POEM (
        POEM_ID integer PRIMARY KEY AUTOINCREMENT,
        CLIENT_ID varchar(20),
        img BLOB NOT NULL,
        POEM varchar(200),
        FEEDBACK int,
        LastCreated DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(CLIENT_ID)
        REFERENCES CLINET(ID)
    );
    """
    )
    conn.commit()


if __name__ == "__main__":
    create_table()
