# RockPhysicsML Windows build and installation

This project includes the five example datasets. Users may still browse for other compatible CSV and NPY datasets.

## Build on Windows

Open a PowerShell terminal in the RockPhysicsML directory, with your working Python virtual environment active.

```powershell
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name RockPhysics --icon oil-pumpjack-svgrepo-com.ico --add-data "oil-pumpjack-svgrepo-com.ico;." gui.py
```

The icon is embedded in the EXE and also bundled for the Tkinter window icon. The `data` folder is **not** embedded in the EXE: Inno Setup installs it beside the EXE so it remains available as shared, read-only sample data.

In Inno Setup Compiler, open `RockPhysicsML_Setup.iss` and select **Build > Compile**. The script expects `dist\\RockPhysics.exe` and `data` in the same project directory as the `.iss` file. The resulting installer is written to the sibling `Installer` directory (e.g., `C:\\RockPhysics_ML\\Installer` if the project is directly under `C:\\RockPhysics_ML`). If your project is nested differently, adjust `OutputDir`.

## File locations

- All-users installation: `C:\Program Files\RockPhysicsML\RockPhysics.exe` (requires administrator approval)
- Shared read-only sample CSV: `C:\Program Files\RockPhysicsML\data\synthetic_wells_combined.csv`
- Shared real-well NPY examples: `C:\Program Files\RockPhysicsML\data\real\`
- Save Plot As dialogs default to the current user's Documents folder. Results batch exports also default there.
- The splash screen uses the included pumpjack icon, displays Version 1.0.2, and credits Joel R. Serfoss. The version is defined once as `APP_VERSION` in `gui.py`.

## Test checklist

1. Back up the working Milestone 3 project and build the new EXE.
2. Test `dist\\RockPhysics.exe` and verify the pumpjack icon in File Explorer, window title bar, and taskbar.
3. Compile and install the setup EXE; launch from Start Menu.
4. Verify the default synthetic CSV loads, training runs, and R²/RMSE/MAE/MAPE appear.
5. Evaluate BT2, SR, F1B and 19A from the Real well evaluation tab.
6. Export metrics and plots, then verify the output files.
7. Browse for another compatible CSV and NPY file outside the installation directory.
8. Test on another compatible Windows 64-bit PC without Python installed. Windows may show an unsigned-app warning.

**Note:** This project has not been executed or installer-tested on Windows in the assistant environment. The commands and paths above must be verified on your machine.
