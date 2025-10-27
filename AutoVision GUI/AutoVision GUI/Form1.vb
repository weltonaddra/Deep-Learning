Public Class Form1
    Private Sub Button1_Click(sender As Object, e As EventArgs) Handles Button1.Click

        'Based on WareData's design for a listbox
        'https://www.youtube.com/watch?v=d69eKagfGK4&t=208s

        If Me.OpenFileDialog1.ShowDialog() = DialogResult.OK Then
            Dim image As String = OpenFileDialog1.FileName
            PictureBox1.ImageLocation = image
            PictureBox1.Refresh()
            Label2.Text = image
            Label2.Visible = True
            Button1.Visible = False
            Button2.Visible = True
            Button3.Visible = True
        Else
            ' Do if user cancel/close dialog
        End If


    End Sub

    Private Sub Button3_Click(sender As Object, e As EventArgs) Handles Button3.Click
        If Me.OpenFileDialog1.ShowDialog() = DialogResult.OK Then
            Dim image As String = OpenFileDialog1.FileName
            PictureBox1.ImageLocation = image
            PictureBox1.Refresh()
            Label2.Text = image
            Label2.Refresh()
        Else
            ' Do if user cancel/close dialog
        End If
    End Sub
End Class
