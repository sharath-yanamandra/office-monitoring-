#!/usr/bin/env python3
"""
Update RTSP URLs in database
"""

import mysql.connector

def update_rtsp_urls():
    """Update camera RTSP URLs"""
    
    # New RTSP URL
    new_url = "rtsp://admin:password@192.168.0.114:554/ch0_0.264"
    
    try:
        # Connect to database
        conn = mysql.connector.connect(
            host='34.93.87.255',
            user='insighteye',
            password='insighteye0411',
            database='dc_test',
            port=3306
        )
        
        cursor = conn.cursor()
        
        # Show current URLs
        print("=== CURRENT CAMERA URLS ===")
        cursor.execute("SELECT camera_id, name, stream_url FROM cameras WHERE datacenter_id = 3")
        cameras = cursor.fetchall()
        for cam in cameras:
            print(f"Camera {cam[0]}: {cam[1]} -> {cam[2]}")
        
        # Update URLs
        print(f"\n🔄 Updating all cameras to: {new_url}")
        cursor.execute("""
            UPDATE cameras 
            SET stream_url = %s 
            WHERE datacenter_id = 3
        """, (new_url,))
        
        conn.commit()
        print(f"✅ Updated {cursor.rowcount} cameras successfully!")
        
        # Show updated URLs
        print("\n=== UPDATED CAMERA URLS ===")
        cursor.execute("SELECT camera_id, name, stream_url FROM cameras WHERE datacenter_id = 3")
        cameras = cursor.fetchall()
        for cam in cameras:
            print(f"Camera {cam[0]}: {cam[1]} -> {cam[2]}")
        
        cursor.close()
        conn.close()
        print("\n🎉 RTSP URLs updated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    update_rtsp_urls()