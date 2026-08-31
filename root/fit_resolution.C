// ---------------------------------------------------------------------------
//  fit_resolution.C
//
//  Quattro fit di  sigma/E = N/E (+) S/sqrt(E) (+) C, salvati come TGraphErrors,
//  TF1 e TCanvas in un file .root versionabile.
//
//  Due insiemi di punti, letti da plot/root/points.csv (li scrive
//  plot/fit_root_all.py):
//     nopos   (sigma/E)^2 = (sigma/mu)^2 - BES^2 - sincrotrone^2
//     pos     come sopra, meno anche POS_eff^2, la sistematica sul centroide
//  In entrambi l'errore e' stat (+) drift.
//
//  Due modi di fittare, per ciascun insieme:
//     indep   N, S, C liberi per ogni resistenza. A 500 ohm C e' fissato a 0.300 %,
//             perche' i dati arrivano solo a 150 GeV e non lo vincolano.
//     common  S e C comuni alle tre resistenze, N libero per resistenza. Realizzato
//             con un TGraphErrors combinato in cui l'ascissa e' E + 10000*indice
//             della resistenza, e una TF1 a 5 parametri che la decodifica: cosi' il
//             fit simultaneo si fa con un normale TGraphErrors::Fit.
//
//  Oggetti scritti, con ds = nopos | pos e R = 340 | 400 | 500:
//     gr_<ds>_<R>              punti
//     f_<ds>_<R>_indep         fit per resistenza
//     f_<ds>_<R>_common        curva del fit comune per quella resistenza
//     gcomb_<ds>, fcomb_<ds>   grafico e funzione del fit simultaneo
//     c_<ds>_indep, c_<ds>_common   canvas a tre pad
//     summary                  TTree coi parametri di tutti i fit
//
//  Uso, dalla radice del repo:
//      root -l -b -q plot/root/fit_resolution.C
//      root -l -b -q 'plot/root/fit_resolution.C("altro.csv","altro.root")'
// ---------------------------------------------------------------------------

#include <TGraphErrors.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TTree.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TAxis.h>
#include <TMath.h>
#include <TStyle.h>
#include <TString.h>
#include <TVirtualPad.h>
#include <TROOT.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <iostream>

const double C_FIX_500 = 0.300;   // C fissato a 500 ohm, come in Python
const double XOFF = 10000.0;      // offset che codifica la resistenza nel grafico combinato

// sigma/E in PERCENTO. N in GeV, S e C in percento: convenzione di fit_plot.sh.
double resoFun(double *x, double *p)
{
   double E = x[0];
   double n = 100.0 * p[0] / E;
   double s = p[1] / TMath::Sqrt(E);
   return TMath::Sqrt(n * n + s * s + p[2] * p[2]);
}

// p[0] = S, p[1] = C, p[2..4] = N di 340, 400, 500 ohm
double resoCombined(double *x, double *p)
{
   int idx = (int)(x[0] / XOFF);
   if (idx < 0) idx = 0;
   if (idx > 2) idx = 2;
   double E = x[0] - XOFF * idx;
   double n = 100.0 * p[2 + idx] / E;
   double s = p[0] / TMath::Sqrt(E);
   return TMath::Sqrt(n * n + s * s + p[1] * p[1]);
}

struct Points {
   std::vector<double> x, y, ex, ey;
};

void fit_resolution(const char *csvname = "plot/root/points.csv",
                    const char *outname = "plot/root/resolution_fits.root")
{
   gStyle->SetOptStat(0);
   gStyle->SetOptFit(0);

   // ------------------------------------------------------------ lettura CSV
   std::ifstream in(csvname);
   if (!in.is_open()) {
      std::cout << "non riesco ad aprire " << csvname << std::endl;
      return;
   }
   std::map<std::string, Points> data;
   std::string line;
   std::getline(in, line);
   while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string ds, sR, sEn, sEt, sY, sE;
      std::getline(ss, ds,  ',');
      std::getline(ss, sR,  ',');
      std::getline(ss, sEn, ',');
      std::getline(ss, sEt, ',');
      std::getline(ss, sY,  ',');
      std::getline(ss, sE,  ',');
      if (sE.empty()) continue;
      Points &p = data[ds + "_" + sR];
      p.x.push_back(std::atof(sEt.c_str()));
      p.y.push_back(std::atof(sY.c_str()));
      p.ex.push_back(0.0);
      p.ey.push_back(std::atof(sE.c_str()));
   }
   in.close();
   if (data.empty()) {
      std::cout << "nessun punto letto da " << csvname << std::endl;
      return;
   }

   TFile *fout = TFile::Open(outname, "RECREATE");
   if (fout == 0 || fout->IsZombie()) {
      std::cout << "non riesco a creare " << outname << std::endl;
      return;
   }

   int    t_ds = 0, t_mode = 0, t_R = 0, t_ndf = 0, t_npoints = 0;
   double t_N = 0, t_eN = 0, t_S = 0, t_eS = 0, t_C = 0, t_eC = 0, t_chi2 = 0;
   TTree *tree = new TTree("summary", "fit N/E (+) S/sqrt(E) (+) C");
   tree->Branch("dataset", &t_ds, "dataset/I");        // 0 = nopos, 1 = pos
   tree->Branch("mode", &t_mode, "mode/I");            // 0 = indep, 1 = common
   tree->Branch("resistance", &t_R, "resistance/I");
   tree->Branch("N_MeV", &t_N, "N_MeV/D");
   tree->Branch("err_N_MeV", &t_eN, "err_N_MeV/D");
   tree->Branch("S_pct", &t_S, "S_pct/D");
   tree->Branch("err_S_pct", &t_eS, "err_S_pct/D");
   tree->Branch("C_pct", &t_C, "C_pct/D");
   tree->Branch("err_C_pct", &t_eC, "err_C_pct/D");
   tree->Branch("chi2", &t_chi2, "chi2/D");
   tree->Branch("ndf", &t_ndf, "ndf/I");
   tree->Branch("npoints", &t_npoints, "npoints/I");

   const int   nds = 2;
   const char *dsname[nds] = {"nopos", "pos"};
   const int   nres = 3;
   const int   resist[nres] = {340, 400, 500};
   const int   colr[nres] = {kAzure + 2, kOrange + 7, kGreen + 2};

   std::cout << std::endl;
   printf("%7s %7s %5s %17s %17s %17s %14s\n",
          "dataset", "mode", "R", "N (MeV)", "S (%)", "C (%)", "chi2/ndf");

   for (int id = 0; id < nds; ++id) {

      // ------------------------------------------------ grafici per resistenza
      TGraphErrors *g[nres];
      bool have[nres];
      for (int ir = 0; ir < nres; ++ir) {
         std::string key = std::string(dsname[id]) + "_" + std::to_string(resist[ir]);
         have[ir] = (data.find(key) != data.end());
         g[ir] = 0;
         if (!have[ir]) continue;
         Points &p = data[key];
         g[ir] = new TGraphErrors((int)p.x.size(), &p.x[0], &p.y[0], &p.ex[0], &p.ey[0]);
         g[ir]->SetName(Form("gr_%s_%d", dsname[id], resist[ir]));
         g[ir]->SetTitle(Form("%d #Omega   %s;True beam energy [GeV];#sigma/E [%%]",
                              resist[ir], dsname[id]));
         g[ir]->SetMarkerStyle(20);
         g[ir]->SetMarkerSize(1.1);
         g[ir]->SetMarkerColor(colr[ir]);
         g[ir]->SetLineColor(colr[ir]);
      }

      // --------------------------------------------------- fit per resistenza
      TF1 *find[nres];
      for (int ir = 0; ir < nres; ++ir) {
         find[ir] = 0;
         if (!have[ir]) continue;
         Points &p = data[std::string(dsname[id]) + "_" + std::to_string(resist[ir])];
         double xmin = *std::min_element(p.x.begin(), p.x.end());
         double xmax = *std::max_element(p.x.begin(), p.x.end());
         TF1 *f = new TF1(Form("f_%s_%d_indep", dsname[id], resist[ir]), resoFun,
                          0.9 * xmin, 1.05 * xmax, 3);
         f->SetParNames("N_GeV", "S_pct", "C_pct");
         f->SetParameters(0.3, 3.0, 0.3);
         f->SetParLimits(0, 0.0, 5.0);
         f->SetParLimits(1, 0.0, 20.0);
         f->SetParLimits(2, 0.0, 5.0);
         if (resist[ir] == 500) f->FixParameter(2, C_FIX_500);
         f->SetLineColor(kViolet);
         f->SetLineStyle(2);
         f->SetLineWidth(3);
         g[ir]->Fit(f, "RQ0");
         find[ir] = f;
      }

      // ------------------------------------------------------- fit simultaneo
      std::vector<double> cx, cy, cex, cey;
      for (int ir = 0; ir < nres; ++ir) {
         if (!have[ir]) continue;
         Points &p = data[std::string(dsname[id]) + "_" + std::to_string(resist[ir])];
         for (size_t k = 0; k < p.x.size(); ++k) {
            cx.push_back(p.x[k] + XOFF * ir);
            cy.push_back(p.y[k]);
            cex.push_back(0.0);
            cey.push_back(p.ey[k]);
         }
      }
      TGraphErrors *gc = new TGraphErrors((int)cx.size(), &cx[0], &cy[0], &cex[0], &cey[0]);
      gc->SetName(Form("gcomb_%s", dsname[id]));
      gc->SetTitle(Form("combined %s;E + 10000 #times index;#sigma/E [%%]", dsname[id]));
      TF1 *fc = new TF1(Form("fcomb_%s", dsname[id]), resoCombined, 0.0, 3.0 * XOFF, 5);
      fc->SetParNames("S_pct", "C_pct", "N340_GeV", "N400_GeV", "N500_GeV");
      fc->SetParameters(2.5, 0.35, 0.30, 0.30, 0.25);
      for (int k = 0; k < 5; ++k) fc->SetParLimits(k, 0.0, (k < 2 ? 20.0 : 5.0));
      gc->Fit(fc, "RQ0");
      double Scom = fc->GetParameter(0), eScom = fc->GetParError(0);
      double Ccom = fc->GetParameter(1), eCcom = fc->GetParError(1);
      double chi2com = fc->GetChisquare();
      int    ndfcom = fc->GetNDF();

      // curve del fit comune, una per resistenza, per poterle disegnare
      TF1 *fcom[nres];
      for (int ir = 0; ir < nres; ++ir) {
         fcom[ir] = 0;
         if (!have[ir]) continue;
         Points &p = data[std::string(dsname[id]) + "_" + std::to_string(resist[ir])];
         double xmin = *std::min_element(p.x.begin(), p.x.end());
         double xmax = *std::max_element(p.x.begin(), p.x.end());
         TF1 *f = new TF1(Form("f_%s_%d_common", dsname[id], resist[ir]), resoFun,
                          0.9 * xmin, 1.05 * xmax, 3);
         f->SetParNames("N_GeV", "S_pct", "C_pct");
         f->FixParameter(0, fc->GetParameter(2 + ir));
         f->FixParameter(1, Scom);
         f->FixParameter(2, Ccom);
         f->SetLineColor(kRed + 1);
         f->SetLineWidth(3);
         fcom[ir] = f;
      }

      // ----------------------------------------------------------- disegno
      for (int mode = 0; mode < 2; ++mode) {
         const char *mname = (mode == 0 ? "indep" : "common");
         TCanvas *c = new TCanvas(Form("c_%s_%s", dsname[id], mname),
                                  Form("%s -- %s", dsname[id], mname), 1800, 600);
         c->Divide(3, 1);
         for (int ir = 0; ir < nres; ++ir) {
            if (!have[ir]) continue;
            TF1 *f = (mode == 0 ? find[ir] : fcom[ir]);
            c->cd(ir + 1);
            gPad->SetLogx();
            gPad->SetGridx();
            gPad->SetGridy();
            g[ir]->Draw("AP");
            f->Draw("same");

            TLegend *leg = new TLegend(0.40, 0.74, 0.90, 0.88);
            leg->SetBorderSize(1);
            leg->SetFillColor(0);
            leg->AddEntry(g[ir], "#sigma/E", "lep");
            leg->AddEntry(f, (mode == 0 ? "N, S, C per resistance"
                                        : "S, C common"), "l");
            leg->Draw();

            TPaveText *pt = new TPaveText(0.40, 0.52, 0.90, 0.73, "NDC");
            pt->SetFillColor(0);
            pt->SetTextAlign(12);
            pt->SetTextFont(82);
            pt->SetTextSize(0.032);
            // in modalita' common la TF1 ha i parametri congelati, quindi l'errore
            // su N va preso dal fit simultaneo, non da lei
            double eN = (mode == 0 ? f->GetParError(0) : fc->GetParError(2 + ir));
            pt->AddText(Form("N %6.1f #pm %4.1f MeV", 1000 * f->GetParameter(0), 1000 * eN));
            pt->AddText(Form("S %6.3f  C %6.4f%s", f->GetParameter(1), f->GetParameter(2),
                             (mode == 0 && resist[ir] == 500) ? "  (C fixed)" : ""));
            if (mode == 0)
               pt->AddText(Form("chi2/ndf %6.2f / %d", f->GetChisquare(), f->GetNDF()));
            else
               pt->AddText(Form("chi2/ndf %6.2f / %d  (global)", chi2com, ndfcom));
            pt->Draw();
         }
         c->Update();
         fout->cd();
         c->Write();
      }

      // ------------------------------------------------- scrittura e stampa
      fout->cd();
      gc->Write();
      fc->Write();
      for (int ir = 0; ir < nres; ++ir) {
         if (!have[ir]) continue;
         g[ir]->Write();
         find[ir]->Write();
         fcom[ir]->Write();

         Points &p = data[std::string(dsname[id]) + "_" + std::to_string(resist[ir])];
         for (int mode = 0; mode < 2; ++mode) {
            TF1 *f = (mode == 0 ? find[ir] : fcom[ir]);
            t_ds = id; t_mode = mode; t_R = resist[ir];
            t_N = 1000 * f->GetParameter(0);
            t_eN = 1000 * (mode == 0 ? f->GetParError(0) : fc->GetParError(2 + ir));
            t_S = f->GetParameter(1);
            t_eS = (mode == 0 ? f->GetParError(1) : eScom);
            t_C = f->GetParameter(2);
            t_eC = (mode == 0 ? f->GetParError(2) : eCcom);
            t_chi2 = (mode == 0 ? f->GetChisquare() : chi2com);
            t_ndf = (mode == 0 ? f->GetNDF() : ndfcom);
            t_npoints = (int)p.x.size();
            tree->Fill();
            printf("%7s %7s %5d %9.1f +- %5.1f %9.4f +- %4.4f %9.5f +- %6.5f %8.2f / %-3d\n",
                   dsname[id], (mode == 0 ? "indep" : "common"), resist[ir],
                   t_N, t_eN, t_S, t_eS, t_C, t_eC, t_chi2, t_ndf);
         }
      }
   }

   fout->cd();
   tree->Write();
   fout->Close();
   std::cout << std::endl << "scritto " << outname << std::endl;
}
