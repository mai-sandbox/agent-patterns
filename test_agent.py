try:
    from agent import app
    print('SUCCESS: Agent imports and app is available')
    print(f'App type: {type(app)}')
except ImportError as e:
    print(f'IMPORT ERROR: {e}')
except AttributeError as e:
    print(f'ATTRIBUTE ERROR: {e}')
except Exception as e:
    print(f'OTHER ERROR: {e}')
