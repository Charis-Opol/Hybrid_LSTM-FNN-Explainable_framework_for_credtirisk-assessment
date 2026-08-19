# Flutter client

Install the Flutter SDK, then run:

```powershell
cd flutter_app
flutter pub get
flutter run
```

For an Android emulator, the default API address `http://10.0.2.2:8000` reaches
the backend on the host machine. For a physical device, replace it with the host
computer’s LAN address, for example `http://192.168.1.10:8000`.

The app accepts an uploaded CSV and borrower ID rather than storing mobile-money
data on-device. The backend stores only the assessment summary locally in SQLite.
