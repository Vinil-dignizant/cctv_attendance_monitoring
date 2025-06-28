; setup.iss
[Setup]
AppName=Face Recognition System
AppVersion=1.0
DefaultDirName={pf}\FaceRecognition
DefaultGroupName=Face Recognition
OutputDir=output
OutputBaseFilename=FaceRecognition_Setup
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\FaceRecognitionSystem.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "config\*"; DestDir: "{app}\config"; Flags: ignoreversion recursesubdirs
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Face Recognition"; Filename: "{app}\FaceRecognitionSystem.exe"
Name: "{commondesktop}\Face Recognition"; Filename: "{app}\FaceRecognitionSystem.exe"

[Run]
Filename: "{app}\FaceRecognitionSystem.exe"; Description: "Launch Application"; Flags: postinstall nowait skipifsilent