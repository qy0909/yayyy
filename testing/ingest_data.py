"""
Data Ingestion Script for Multilingual RAG Bot
===============================================
This script generates embeddings for sample documents and uploads them to Supabase.

Usage:
    python ingest_data.py

Requirements:
    - .env file with SUPABASE_URL and SUPABASE_KEY
    - Supabase table 'embeddings' created (run supabase_setup.sql first)
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import numpy as np
from typing import List, Dict
import time

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
EMBEDDINGS_TABLE_NAME = 'embeddings'

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
print(f"✓ Embedding model loaded (dimension: {embedding_model.get_sentence_embedding_dimension()})")

# Initialize Supabase client
print(f"Connecting to Supabase at {SUPABASE_URL}...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✓ Connected to Supabase")


# ============================================================================
# Sample Documents for Malaysian Migrant Workers
# ============================================================================

SAMPLE_DOCUMENTS = [
    # Healthcare - Malay
    {
        "content": "Pekerja asing di Malaysia layak mendapat subsidi penjagaan kesihatan. Anda perlu bawa passport, permit kerja, dan bukti pendapatan ke klinik kesihatan berhampiran. Rawatan untuk penyakit biasa dikenakan bayaran RM1 hingga RM5 sahaja. Untuk kecemasan, pergi ke Hospital Kerajaan terdekat.",
        "title": "Subsidi Kesihatan untuk Pekerja Asing",
        "source_url": "https://www.moh.gov.my/healthcare-subsidy",
        "language": "Malay",
        "document_type": "healthcare",
        "region": "Malaysia"
    },
    {
        "content": "Klinik panel majikan anda menyediakan rawatan percuma atau subsidi. Tanya bahagian HR syarikat untuk senarai klinik panel. Anda juga boleh mendapatkan ubat-ubatan asas di farmasi kerajaan dengan harga murah. Bawa salinan permit kerja anda.",
        "title": "Klinik Panel Majikan",
        "source_url": "https://www.moh.gov.my/panel-clinics",
        "language": "Malay",
        "document_type": "healthcare",
        "region": "Malaysia"
    },
    
    # Healthcare - Tagalog
    {
        "content": "Ang mga dayuhang manggagawa sa Malaysia ay may karapatan sa subsidized healthcare. Dalhin ang iyong passport, work permit, at proof of income sa pinakamalapit na government clinic. Ang bayad para sa karaniwang sakit ay RM1 hanggang RM5 lamang. Para sa emergency, pumunta sa pinakamalapit na Government Hospital.",
        "title": "Healthcare Subsidy para sa Foreign Workers",
        "source_url": "https://www.moh.gov.my/healthcare-subsidy-tagalog",
        "language": "Tagalog",
        "document_type": "healthcare",
        "region": "Malaysia"
    },
    {
        "content": "May libre o subsidized na serbisyo sa panel clinic ng iyong employer. Magtanong sa HR department para sa listahan ng panel clinics. Maaari ka ring bumili ng murang gamot sa government pharmacy. Magdala ng kopya ng iyong work permit.",
        "title": "Panel Clinic ng Employer",
        "source_url": "https://www.moh.gov.my/panel-clinics-tagalog",
        "language": "Tagalog",
        "document_type": "healthcare",
        "region": "Malaysia"
    },
    
    # Immigration - Malay
    {
        "content": "Untuk memperbaharui permit kerja di Malaysia, datang ke Jabatan Imigresen dengan membawa passport asal, laporan perubatan terkini, dan surat penajaan majikan. Proses mengambil masa 2-4 minggu. Bayaran pemprosesan adalah RM100-RM500 bergantung kepada jenis permit. Pastikan permit sedia ada belum tamat tempoh lebih daripada 30 hari.",
        "title": "Pembaharuan Permit Kerja",
        "source_url": "https://www.imi.gov.my/permit-renewal",
        "language": "Malay",
        "document_type": "immigration",
        "region": "Malaysia"
    },
    {
        "content": "Pekerja asing mesti daftar dengan FOMEMA untuk pemeriksaan perubatan. Kos adalah kira-kira RM200-RM300. Lawati klinik FOMEMA berdaftar dengan appointment. Hasil pemeriksaan akan dihantar terus ke Jabatan Imigresen. Tanpa peperiksaan FOMEMA yang sah, permit kerja tidak akan diluluskan.",
        "title": "Pemeriksaan Perubatan FOMEMA",
        "source_url": "https://www.fomema.com.my",
        "language": "Malay",
        "document_type": "immigration",
        "region": "Malaysia"
    },
    
    # Immigration - Tagalog
    {
        "content": "Para sa pagpapanibago ng work permit sa Malaysia, pumunta sa Immigration Department may dalang original passport, updated medical report, at employer sponsorship letter. Ang proseso ay tumatagal ng 2-4 linggo. Ang bayad sa processing ay RM100-RM500 depende sa uri ng permit. Siguraduhing hindi pa nag-eexpire ang kasalukuyang permit ng mahigit 30 araw.",
        "title": "Pagpapanibago ng Work Permit",
        "source_url": "https://www.imi.gov.my/permit-renewal-tagalog",
        "language": "Tagalog",
        "document_type": "immigration",
        "region": "Malaysia"
    },
    {
        "content": "Kailangan ng lahat ng foreign workers na magrehistro sa FOMEMA para sa medical examination. Ang halaga ay humigit-kumulang RM200-RM300. Bumisita sa registered FOMEMA clinic with appointment. Ang resulta ay direktang ipapadala sa Immigration Department. Walang valid FOMEMA exam, hindi aaprubahan ang work permit.",
        "title": "FOMEMA Medical Examination",
        "source_url": "https://www.fomema.com.my/tagalog",
        "language": "Tagalog",
        "document_type": "immigration",
        "region": "Malaysia"
    },
    
    # Housing - English
    {
        "content": "Foreign workers in Penang can find affordable housing through employer-provided accommodation or private rentals. Monthly rent ranges from RM200-RM600 per person in shared housing. Check that the property has valid rental license from local council. Employers are required to provide adequate housing or housing allowance as per labor laws.",
        "title": "Housing Options for Foreign Workers in Penang",
        "source_url": "https://penang.gov.my/foreign-worker-housing",
        "language": "English",
        "document_type": "housing",
        "region": "Penang"
    },
    {
        "content": "Workers' minimum standards for accommodation in Malaysia: At least 50 square feet per person, proper ventilation, clean water supply, functioning toilets (1 per 15 persons), and basic furniture. Report substandard housing to Department of Labour. Free legal aid available through Tenaganita and other NGOs.",
        "title": "Minimum Housing Standards",
        "source_url": "https://www.mohr.gov.my/housing-standards",
        "language": "English",
        "document_type": "housing",
        "region": "Malaysia"
    },
    
    # Labor Rights - English
    {
        "content": "Foreign workers in Malaysia are entitled to: minimum wage (RM1,500/month), overtime pay (1.5x hourly rate), rest days (1 day per week), annual leave (8-16 days), sick leave with medical certificate, and public holidays. Contact Department of Labour for violations: 1-800-88-8777. All complaints are confidential.",
        "title": "Foreign Worker Rights in Malaysia",
        "source_url": "https://www.mohr.gov.my/worker-rights",
        "language": "English",
        "document_type": "labor_rights",
        "region": "Malaysia"
    },
    {
        "content": "If your employer withholds your passport, delays salary payment, or forces you to work excessive hours, these are labor law violations. File a complaint with Department of Labour online or visit nearest office. Legal aid is available through Legal Aid Centre (1-800-88-1548). You cannot be deported for making complaints.",
        "title": "Reporting Labor Violations",
        "source_url": "https://www.mohr.gov.my/report-violations",
        "language": "English",
        "document_type": "labor_rights",
        "region": "Malaysia"
    },
    
    # Financial Services - Malay
    {
        "content": "Pekerja asing boleh membuka akaun bank asas di Malaysia. Bawa passport, permit kerja sah, dan surat pengesahan majikan ke bank-bank utama seperti Maybank, CIMB, atau Bank Islam. Tiada bayaran penyelenggaraan bulanan untuk akaun asas. Anda boleh menggunakan perkhidmatan pemindahan wang seperti Western Union atau remittance agents berlesen.",
        "title": "Perkhidmatan Perbankan untuk Pekerja Asing",
        "source_url": "https://www.bnm.gov.my/banking-foreign-workers",
        "language": "Malay",
        "document_type": "financial",
        "region": "Malaysia"
    },
    
    # Financial Services - Tagalog
    {
        "content": "Ang mga foreign workers ay maaaring magbukas ng basic bank account sa Malaysia. Dalhin ang passport, valid work permit, at employer confirmation letter sa major banks tulad ng Maybank, CIMB, o Bank Islam. Walang monthly maintenance fee para sa basic accounts. Maaari mong gamitin ang money transfer services tulad ng Western Union o licensed remittance agents.",
        "title": "Banking Services para sa Foreign Workers",
        "source_url": "https://www.bnm.gov.my/banking-foreign-workers-tagalog",
        "language": "Tagalog",
        "document_type": "financial",
        "region": "Malaysia"
    },
    
    # Emergency Contacts - Mixed (English with phone numbers)
    {
        "content": "Important emergency contacts for foreign workers in Malaysia: Police/Emergency 999, Ambulance 999, Fire Department 994, Department of Labour 1-800-88-8777, Immigration Department 03-8000-8000, Philippine Embassy KL 03-2148-4848, Indonesian Embassy KL 03-2116-4000, Bangladesh High Commission 03-2161-1088. Save these numbers in your phone.",
        "title": "Emergency Contact Numbers",
        "source_url": "https://www.malaysia.gov.my/emergency-contacts",
        "language": "English",
        "document_type": "emergency",
        "region": "Malaysia"
    },
    
    # COVID-19 / Health Emergency - Malay
    {
        "content": "Pekerja asing layak mendapat vaksin COVID-19 percuma di Malaysia. Daftar melalui MySejahtera app atau lawati PPV (Pusat Pemberian Vaksin) terdekat dengan passport dan permit kerja. Jika sakit, hubungi Crisis Preparedness and Response Centre (CPRC) COVID-19 di talian 03-8881-0200 atau 03-8881-0600. Kuarantin dan rawatan adalah percuma.",
        "title": "Vaksinasi COVID-19 untuk Pekerja Asing",
        "source_url": "https://www.moh.gov.my/covid-vaccine",
        "language": "Malay",
        "document_type": "healthcare",
        "region": "Malaysia"
    }
]


# ============================================================================
# Helper Functions
# ============================================================================

def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for given text."""
    embedding = embedding_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def insert_document(doc: Dict) -> bool:
    """Insert a single document with its embedding into Supabase."""
    try:
        # Generate embedding
        embedding = generate_embedding(doc['content'])
        
        # Prepare data for insertion
        data = {
            'content': doc['content'],
            'title': doc['title'],
            'source_url': doc['source_url'],
            'language': doc['language'],
            'document_type': doc['document_type'],
            'region': doc['region'],
            'embedding': embedding
        }
        
        # Insert into Supabase
        result = supabase.table(EMBEDDINGS_TABLE_NAME).insert(data).execute()
        
        return True
    except Exception as e:
        print(f"  ✗ Error inserting document '{doc['title']}': {e}")
        return False


def batch_insert_documents(documents: List[Dict], batch_size: int = 5) -> Dict:
    """Insert multiple documents in batches."""
    total = len(documents)
    success_count = 0
    fail_count = 0
    
    print(f"\nInserting {total} documents...")
    print("-" * 80)
    
    for i, doc in enumerate(documents, 1):
        print(f"[{i}/{total}] Processing: {doc['title'][:50]}...")
        print(f"         Language: {doc['language']}, Type: {doc['document_type']}")
        
        if insert_document(doc):
            success_count += 1
            print(f"         ✓ Inserted successfully")
        else:
            fail_count += 1
        
        # Small delay to avoid rate limits
        if i % batch_size == 0:
            time.sleep(0.5)
    
    print("-" * 80)
    print(f"\nIngestion complete!")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    
    return {
        'total': total,
        'success': success_count,
        'failed': fail_count
    }


def verify_data() -> None:
    """Verify that data was inserted correctly."""
    print("\n" + "=" * 80)
    print("VERIFYING DATA")
    print("=" * 80)
    
    try:
        # Count total documents
        result = supabase.table(EMBEDDINGS_TABLE_NAME).select('id', count='exact').execute()
        total_count = result.count
        print(f"\n✓ Total documents in database: {total_count}")
        
        # Count by language
        languages = ['English', 'Malay', 'Tagalog']
        for lang in languages:
            result = supabase.table(EMBEDDINGS_TABLE_NAME).select('id', count='exact').eq('language', lang).execute()
            print(f"  - {lang}: {result.count} documents")
        
        # Count by document type
        doc_types = ['healthcare', 'immigration', 'housing', 'labor_rights', 'financial', 'emergency']
        print("\nDocuments by type:")
        for dtype in doc_types:
            result = supabase.table(EMBEDDINGS_TABLE_NAME).select('id', count='exact').eq('document_type', dtype).execute()
            if result.count > 0:
                print(f"  - {dtype}: {result.count} documents")
        
        # Sample a few documents
        print("\nSample documents:")
        result = supabase.table(EMBEDDINGS_TABLE_NAME).select('title', 'language', 'document_type').limit(3).execute()
        for doc in result.data:
            print(f"  - [{doc['language']}] {doc['title']} ({doc['document_type']})")
        
        print("\n✓ Data verification complete!")
        
    except Exception as e:
        print(f"\n✗ Error verifying data: {e}")


def test_vector_search() -> None:
    """Test the vector search function."""
    print("\n" + "=" * 80)
    print("TESTING VECTOR SEARCH")
    print("=" * 80)
    
    test_queries = [
        "How do I get healthcare in Malaysia?",
        "Paano ako mag-renew ng work permit?",
        "Klinik mana yang murah?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            # Generate query embedding
            query_embedding = generate_embedding(query)
            print(f"  → Query embedding dimension: {len(query_embedding)}")
            
            # Call match_documents function
            result = supabase.rpc('match_documents', {
                'query_embedding': query_embedding,
                'match_count': 3,
                'similarity_threshold': 0.3
            }).execute()
            
            if result.data:
                print(f"  ✓ Found {len(result.data)} results:")
                for i, doc in enumerate(result.data, 1):
                    print(f"    {i}. [{doc['language']}] {doc['title']}")
                    print(f"       Similarity: {doc['similarity']:.3f}")
            else:
                print("  ! No results found (similarity threshold may be too high)")
                
        except Exception as e:
            print(f"  ✗ Search error: {e}")


def clear_existing_data() -> None:
    """Clear all existing data from the embeddings table."""
    print("\n" + "=" * 80)
    print("CLEARING EXISTING DATA")
    print("=" * 80)
    
    try:
        # Count existing records
        result = supabase.table(EMBEDDINGS_TABLE_NAME).select('id', count='exact').execute()
        existing_count = result.count
        
        if existing_count == 0:
            print("\n✓ Table is already empty")
            return
        
        # Confirm deletion
        print(f"\n⚠️  Found {existing_count} existing documents")
        confirm = input("Delete all existing data? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            # Delete all records
            supabase.table(EMBEDDINGS_TABLE_NAME).delete().neq('id', 0).execute()
            print(f"✓ Deleted {existing_count} documents")
        else:
            print("✗ Deletion cancelled")
            
    except Exception as e:
        print(f"✗ Error clearing data: {e}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main ingestion workflow."""
    print("\n" + "=" * 80)
    print("MULTILINGUAL RAG BOT - DATA INGESTION")
    print("=" * 80)
    
    # Check environment variables
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("\n✗ Error: SUPABASE_URL or SUPABASE_KEY not found in .env file")
        print("  Please update your .env file with valid credentials")
        return
    
    print(f"\n✓ Supabase URL: {SUPABASE_URL}")
    print(f"✓ API Key: {SUPABASE_KEY[:20]}...")
    
    # Menu
    print("\nOptions:")
    print("1. Insert sample documents (keeps existing data)")
    print("2. Clear all data and insert sample documents")
    print("3. Verify existing data")
    print("4. Test vector search")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        # Insert without clearing
        batch_insert_documents(SAMPLE_DOCUMENTS)
        verify_data()
        test_vector_search()
        
    elif choice == '2':
        # Clear and insert
        clear_existing_data()
        batch_insert_documents(SAMPLE_DOCUMENTS)
        verify_data()
        test_vector_search()
        
    elif choice == '3':
        # Verify only
        verify_data()
        
    elif choice == '4':
        # Test search only
        test_vector_search()
        
    elif choice == '5':
        print("\nExiting...")
        return
    else:
        print("\n✗ Invalid option")
        return
    
    print("\n" + "=" * 80)
    print("✓ DATA INGESTION COMPLETE!")
    print("=" * 80)
    print("\nYour RAG pipeline is now ready to use!")
    print("Run: python pipeline/RAG.py")


if __name__ == "__main__":
    main()
