"""
Seed script to populate the local Postgres database with sample tenant data and vector embeddings.
Run this once after starting the app to populate sample data for testing.

Usage: python seed.py
"""

import asyncio
import os
from uuid import UUID
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy.exc import IntegrityError

from database import AsyncSessionLocal
from models import Organization, Document, DocumentChunk

# Load environment variables
load_dotenv()

# Static UUIDs for easy copy-paste during testing
TECHCORP_ORG_ID = UUID("11111111-1111-1111-1111-111111111111")

# Sample support knowledge base chunks
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


async def seed_database():
    """
    Main seeding function. Creates an organization, documents, and vector embeddings.
    """
    print("🌱 Starting database seed...")

    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    async with AsyncSessionLocal() as session:
        try:
            # 1. Check if organization already exists
            org = await session.get(Organization, TECHCORP_ORG_ID)
            if org:
                print(f"✅ Organization '{org.name}' already exists with ID: {org.id}")
            else:
                # Create organization
                org = Organization(
                    id=TECHCORP_ORG_ID,
                    name="TechCorp SaaS",
                )
                session.add(org)
                await session.flush()
                print(f"✅ Created Organization: TechCorp SaaS (ID: {TECHCORP_ORG_ID})")

            # 2. Create documents and chunks with embeddings
            for chunk_idx, chunk_info in enumerate(SAMPLE_CHUNKS):
                # Create document
                doc = Document(
                    organization_id=TECHCORP_ORG_ID,
                    title=chunk_info["title"],
                    source_url=f"https://docs.techcorp.com/{chunk_info['title'].lower().replace(' ', '-')}",
                )
                session.add(doc)
                await session.flush()

                # Generate embedding for the chunk content
                print(f"  📝 Generating embedding for chunk {chunk_idx + 1}: '{chunk_info['title']}'...")
                embedding_vector = await embeddings.aembed_query(chunk_info["content"])

                # Create document chunk with embedding
                doc_chunk = DocumentChunk(
                    document_id=doc.id,
                    content=chunk_info["content"],
                    chunk_index=chunk_idx,
                    embedding=embedding_vector,
                )
                session.add(doc_chunk)

                print(
                    f"  ✅ Saved chunk {chunk_idx + 1}: '{chunk_info['title']}' "
                    f"(768-dim embedding, {len(chunk_info['content'])} chars)"
                )

            # 3. Commit all changes
            await session.commit()
            print("\n✅ Database seeding completed successfully!")
            print(f"\n📊 Summary:")
            print(f"   - Organization: TechCorp SaaS ({TECHCORP_ORG_ID})")
            print(f"   - Documents & Chunks: {len(SAMPLE_CHUNKS)}")
            print(f"   - Total Embeddings: {len(SAMPLE_CHUNKS)} (768-dim vectors)")
            print(f"\n💡 Use this UUID in your Streamlit frontend: {TECHCORP_ORG_ID}")

        except IntegrityError as e:
            await session.rollback()
            print(f"⚠️  Integrity error (likely duplicate data): {str(e)}")
            print("   This is normal if you've already seeded the database.")
        except Exception as e:
            await session.rollback()
            print(f"❌ Error during seeding: {str(e)}")
            raise


if __name__ == "__main__":
    print("🚀 AI Support Operations Platform - Database Seeder")
    print("=" * 60)
    asyncio.run(seed_database())
