Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() | ForEach-Object {
    $info = $_.VoiceInfo
    Write-Output ("{0}  |  culture={1}  |  gender={2}" -f $info.Name, $info.Culture, $info.Gender)
}
$synth.Dispose()
