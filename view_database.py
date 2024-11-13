import sqlite3

# Connect to the database
conn = sqlite3.connect('society_vehicles.db')
cursor = conn.cursor()

# Query to select all data from the vehicle_owners table
cursor.execute("SELECT * FROM vehicle_owners")
rows = cursor.fetchall()

# Print the data
if rows:
    print("Data in vehicle_owners table:")
    for row in rows:
        print(row)
else:
    print("No data found in the vehicle_owners table.")

# Close the connection
conn.close()
