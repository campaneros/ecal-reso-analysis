// Cross-check ROOT vs Python.
//
// Rifa' ESATTAMENTE la ricetta di fit.sh + dcb.cxx su un file merged e stampa
// una riga CSV, poi rifa' il profilo di A_tot vs centroide con gli stessi tagli
// espliciti che usa drift_dcb_all.py e stampa costante e chi2/ndf.
//
// Uso (dalla cartella che contiene dcb.cxx):
//   root -l -b -q 'crosscheck_root.C("reco_400ohm/150_400_merged.root",150,400,"root_fit.csv")'
//
// oppure in blocco:  ./crosscheck_root.sh <cartella_base> <dcb_dir>

void crosscheck_root(const char *file, int en, int resistance,
                     const char *outfit = "root_fit.csv",
                     const char *dcbdir = ".")
{
  double scale = 0;
  if      (resistance == 340) scale = 3500. / 150.;
  else if (resistance == 400) scale = 1080. / 40.;
  else if (resistance == 500) scale = 3340. / 100.;
  else { printf("resistenza sconosciuta\n"); return; }

  gROOT->ProcessLine(Form(".L %s/dcb.cxx", dcbdir));

  TFile *f = TFile::Open(file);
  TTree *t = (TTree *)f->Get("h4_reco");
  if (!t) { printf("h4_reco non trovato in %s\n", file); return; }

  // ---------------------------------------------------------------- fit.sh
  TString hname = Form("FitAmp_3x3_%d_uncalibrated", en);
  t->Draw(Form("A_tot>>%s(8000,0,8000)", hname.Data()),
          "abs(pos_eta-18)<=0.2 && abs(pos_phi-6)<=0.2");
  TH1 *h = (TH1 *)gDirectory->Get(hname);

  h->GetXaxis()->SetRangeUser(scale * en * 0.95, scale * en * 1.05);
  for (int i = 0; i < 2; ++i)
    h->GetXaxis()->SetRangeUser(h->GetMean() - 3 * h->GetRMS(),
                                h->GetMean() + 3 * h->GetRMS());
  for (int i = 0; i < 3; ++i) gROOT->ProcessLine(Form("dcb((TH1*)%p);", (void *)h));

  TF1 *fn = h->GetFunction("dcb");
  double chi2 = fn->GetChisquare();
  int ndf = fn->GetNDF();

  bool newfile = gSystem->AccessPathName(outfit);
  FILE *o = fopen(outfit, "a");
  if (newfile) fprintf(o, "resistance,energy,peak_abs,err_peak_abs,sigma_abs,"
                          "err_sigma_abs,chi2,ndf,nentries\n");
  fprintf(o, "%d,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%d,%.0f\n", resistance, en,
          fn->GetParameter(4), fn->GetParError(4),
          fn->GetParameter(5), fn->GetParError(5), chi2, ndf, h->GetEntries());
  fclose(o);
  printf(">> FIT  %d ohm %d GeV : peak %.4f +- %.4f   sigma %.4f +- %.4f   chi2/ndf %.3f\n",
         resistance, en, fn->GetParameter(4), fn->GetParError(4),
         fn->GetParameter(5), fn->GetParError(5), chi2 / ndf);

  f->Close();
}
