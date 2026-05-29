import os
import pickle

print("Prueba exitosa")

if os.path.exists('model.pkl'):
	try:
		with open('model.pkl', 'rb') as f:
			model = pickle.load(f)
		# try a dummy prediction
		sample = [5.0, 80.0, 5.0, 11.0]
		pred = model.predict([sample])
		print('Smoke prediction OK:', pred)
	except Exception as e:
		print('Smoke prediction failed:', e)
		raise