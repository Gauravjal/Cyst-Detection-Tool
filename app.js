const express = require("express");
const multer = require("multer");
const ejs = require("ejs");
const path = require("path");

// Set Storage Engine
const storage = multer.diskStorage({
  destination: "./public/uploads/",
  filename: function (req, file, cb) {
    cb(null, "test" + ".jpg");
  },
});

//Init Upload
const upload = multer({
  storage: storage,
  //limits:{fileSize:10},
  fileFilter: function (req, file, cb) {
    checkFileType(file, cb);
  },
}).single("myimage");

//check file type
function checkFileType(file, cb) {
  // Allowed ext
  const filetypes = /jpeg|jpg|png|gif/;
  //check ext
  const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
  //check mime
  const mimetype = filetypes.test(file.mimetype);

  if (mimetype && extname) {
    return cb(null, true);
  } else {
    cb("Error Images only");
  }
}

//Init app
const app = express();

//Ejs
app.set("view engine", "ejs");

//public folder
app.use(express.static("./public"));

app.get("/", function (req, res) {
  res.render("index");
});

app.post("/upload", function (req, res) {
  handleUpload(req, res);
});

function handleUpload(req, res) {
  upload(req, res, function (err) {
    if (err instanceof multer.MulterError) {
      console.error(err);
      res.status(500).send("Multer error occurred!");
    } else if (err) {
      console.error(err);
      res.status(500).send("Unknown error occurred!");
    } else {
      console.log("callname " + req.file.filename);
      var spawn = require("child_process").spawn;
      console.log("checkpoint");
      var responseData = {};

      var process = spawn("python3", ["./Generate.py", req.file.filename]);
      process.stdout.on("data", function (data) {
        console.log("output after " + data);
        responseData.msg = "File Uploaded!";
      });

      process.on("close", function () {
        // Send the response after file processing is done
        res.render("index", responseData);
      });
    }
  });
}

const port = 3000;

app.listen(port, function () {
  console.log(`server started on port ${port}`);
});
