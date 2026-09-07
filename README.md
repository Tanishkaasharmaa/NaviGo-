# NaviGo-

## How to run?

1. Create virtual environment
'''bash
python -m venv myenv
'''
2. Activate virtual environment
'''bash
myenv\Scripts\activate.bat
'''
3. Install requirements
'''bash
pip install -r requirements.txt
'''
4. Run the application
'''bash
python app.py
'''

## Environment variables

Create a .env file in the root directory with the following variables:
'''bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=your_database_url
'''


DATABASE_URL=postgresql://tanishkasharma:ImfVvHNm16aPl3m67nyysUCEDIYPJWBn@dpg-daeoie2d0e5s739c5j20-a.ohio-postgres.render.com/navigo