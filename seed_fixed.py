"""
Sync seed script using psycopg2 directly - bypasses asyncpg vector type issues.
Usage: python seed_sync.py
"""

import os
import psycopg2
from uuid import UUID, uuid4
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

TECHCORP_ORG_ID = "11111111-1111-1111-1111-111111111111"

SAMPLE_CHUNKS = [
    {
        "title": "Password Policy",
        "content": "TechCorp Password Policy: Passwords must be 12 characters long, include uppercase and lowercase letters, numbers, and special characters. Users can reset via the /reset-password link on the login page. Password reset emails expire after 24 hours.",
    },
    {
        "title": "Refund Policy",
        "content": "TechCorp Refund Policy: We offer a 14-day money-back guarantee from the date of purchase. Software licenses are non-refundable after activation. To request a refund, contact our billing team at billing@techcorp.com with your invoice number. Refunds are processed within 5-7 business days.",
    },
    {
        "title": "API Rate Limits",
        "content": "TechCorp API Rate Limits: Free tier is limited to 100 requests per minute. Pro tier allows 1000 requests per minute. Enterprise tier offers custom limits. Rate limits reset at the start of each calendar month. If you exceed limits, you'll receive a 429 Too Many Requests error.",
    },
]


def get_psycopg2_url():
    """Convert asyncpg URL to psycopg2 URL."""
    url = os.getenv("DATABASE_URL", "")
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("?ssl=require", "")
    return url


def seed_database():
    print("🌱 Starting database seed...")

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    db_url = get_psycopg2_url()
    conn = psycopg2.connect(db_url, sslmode="require")
    cur = conn.cursor()

    try:
        # Check/create organization
        cur.execute("SELECT id FROM organizations WHERE id = %s", (TECHCORP_ORG_ID,))
        org = cur.fetchone()

        if org:
            print(f"✅ Organization already exists.")
        else:
            cur.execute("""
                INSERT INTO organizations (id, name, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW())
            """, (TECHCORP_ORG_ID, "TechCorp SaaS"))
            print(f"✅ Created Organization: TechCorp SaaS")

        for chunk_idx, chunk_info in enumerate(SAMPLE_CHUNKS):
            doc_id = str(uuid4())

            cur.execute("""
                INSERT INTO documents (id, organization_id, title, source_url, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
            """, (
                doc_id,
                TECHCORP_ORG_ID,
                chunk_info["title"],
                f"https://docs.techcorp.com/{chunk_info['title'].lower().replace(' ', '-')}"
            ))

            print(f"  📝 Embedding chunk {chunk_idx + 1}: '{chunk_info['title']}'...")

            embed_result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=chunk_info["content"]
            )
            embedding_vector = embed_result['embedding']
            embedding_str = "[" + ",".join(map(str, embedding_vector)) + "]"

            cur.execute("""
                INSERT INTO document_chunks (id, document_id, content, chunk_index, embedding, created_at, updated_at)
                VALUES (gen_random_uuid(), %s, %s, %s, %s::vector, NOW(), NOW())
            """, (doc_id, chunk_info["content"], chunk_idx, embedding_str))

            print(f"  ✅ Saved chunk {chunk_idx + 1}: '{chunk_info['title']}'")

        conn.commit()
        print("\n✅ Database seeding completed successfully!")
        print(f"   - Organization: TechCorp SaaS ({TECHCORP_ORG_ID})")
        print(f"   - 3 chunks inserted with 3072-dim embeddings")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    print("🚀 AI Support Operations Platform - Database Seeder")
    print("=" * 60)
    seed_database()