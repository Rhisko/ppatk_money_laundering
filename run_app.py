#!/usr/bin/env python3
"""
Script to run the Streamlit Money Laundering Detection App
"""
import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'xgboost', 'lightgbm', 'openpyxl'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def run_streamlit():
    """Run the Streamlit app"""
    if not check_requirements():
        return
    
    print("\n🚀 Starting Money Laundering Detection Streamlit App...")
    print("📊 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*60)
    
    try:
        # Run Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    run_streamlit()
