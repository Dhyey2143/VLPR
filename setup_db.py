import sqlite3

# Connect to (or create) the database file
conn = sqlite3.connect('society_vehicles.db')
cursor = conn.cursor()

# Create a table to store vehicle owner information
cursor.execute('''CREATE TABLE IF NOT EXISTS vehicle_owners (
    owner_name TEXT,
    license_plate TEXT UNIQUE,
    apartment_number TEXT,
    contact_details TEXT
)''')

# Insert sample data into the table (adjust these to match your test images)
sample_data = [
    ('John Doe', 'ABC1234', 'A-101', '+1234567890'),
    ('Jane Smith', 'XYZ5678', 'B-202', '+1234567891'),
    ('Alice Brown', 'LMN9101', 'C-303', '+1234567892'),
    ('Robert White', 'PQR4567', 'D-404', '+1234567893'),
]

# Insert the data, ignoring if it already exists
cursor.executemany('INSERT OR IGNORE INTO vehicle_owners VALUES (?, ?, ?, ?)', sample_data)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database setup completed with sample data.")
